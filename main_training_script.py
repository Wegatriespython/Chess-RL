import torch
import time
import json
import chess
from threading import Thread
from model import ChessValueNetwork
from mcts import Node, mcts_search
from chess_utils import board_to_input

def save_checkpoint(model, optimizer, stats, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, filename)

def gpu_monitor():
    while True:
        gpu = torch.cuda.get_device_properties(0)
        utilization = torch.cuda.utilization(0)
        print(f"GPU Utilization: {utilization}%, Memory: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB / {gpu.total_memory/1024**3:.2f}GB")
        time.sleep(5)

def evaluate_game_result(board):
    if board.is_checkmate():
        return 1 if board.turn == chess.BLACK else -1
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
        return 0
    else:
        # If the game isn't over, we shouldn't be here, but let's return a draw
        return 0

def main(test_mode=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ChessValueNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Hyperparameters
    batch_size = 8 if test_mode else 64
    num_games = 10 if test_mode else 1000
    max_moves_per_game = 100 if test_mode else 200

    stats = {
        'games_played': 0,
        'total_losses': [],
        'avg_game_lengths': [],
    }

    # Start GPU monitoring
    if device.type == 'cuda':
        Thread(target=gpu_monitor, daemon=True).start()
        
    save_interval = 60 if test_mode else 600  # Seconds
    start_time = time.time()
    last_save_time = start_time

    print("Starting training...")
    for game_num in range(num_games):
        board = chess.Board()
        game_moves = []
        game_losses = []

        for move_num in range(max_moves_per_game):
            if board.is_game_over():
                break

            root = Node(board)
            best_move = mcts_search(root, model, board_to_input, device)

            if best_move:
                board.push(best_move)
                game_moves.append((board.copy(), best_move))
            else:
                print(f"No legal moves found in game {game_num + 1}, move {move_num + 1}")
                break

        # Determine game outcome
        evaluation = evaluate_game_result(board)

        # Update model for each move in the game
        for board_state, move in game_moves:
            optimizer.zero_grad()
            board_input = board_to_input(board_state, device).unsqueeze(0)
            predicted_value = model(board_input).view(-1)
            target = torch.tensor([evaluation], device=device, dtype=torch.float32).view(-1)
            loss = torch.nn.MSELoss()(predicted_value, target)
            loss.backward()
            optimizer.step()

            game_losses.append(loss.item())

        avg_game_loss = sum(game_losses) / len(game_losses) if game_losses else 0
        stats['games_played'] += 1
        stats['total_losses'].extend(game_losses)
        stats['avg_game_lengths'].append(len(game_moves))

        print(f"Game {stats['games_played']}, Result: {evaluation:.1f}, Avg Loss: {avg_game_loss:.4f}, Moves: {len(game_moves)}")

        # Save checkpoint
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            save_checkpoint(model, optimizer, stats, f"chess_model_checkpoint_{stats['games_played']}.pth")
            last_save_time = current_time

        elapsed_time = current_time - start_time
        print(f"Time elapsed: {elapsed_time:.2f}s")

        if test_mode and elapsed_time >= 300:  # 5 minutes for test mode
            print("Test mode time limit reached. Ending training.")
            break

    # Save final model and stats
    save_checkpoint(model, optimizer, stats, "chess_model_final.pth")
    with open("training_stats.json", "w") as f:
        json.dump(stats, f)

    print("Training complete. Model and stats saved.")
    print(f"Total games played: {stats['games_played']}")
    print(f"Average game length: {sum(stats['avg_game_lengths']) / len(stats['avg_game_lengths']):.2f} moves")
    print(f"Final average loss: {sum(stats['total_losses'][-1000:]) / min(len(stats['total_losses']), 1000):.4f}")

if __name__ == "__main__":
    main()