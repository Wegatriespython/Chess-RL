import torch
import time
import json
import chess
import random
from collections import deque
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
        return 1.0 if board.turn == chess.BLACK else -1.0
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    elif board.is_fifty_moves() or board.is_repetition():
        return -0.1  # Slight penalty for drawish positions
    else:
        # Heuristic evaluation
        white_material = sum(len(board.pieces(piece_type, chess.WHITE)) * value 
                             for piece_type, value in zip([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN], 
                                                          [1, 3, 3, 5, 9]))
        black_material = sum(len(board.pieces(piece_type, chess.BLACK)) * value 
                             for piece_type, value in zip([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN], 
                                                          [1, 3, 3, 5, 9]))
        return (white_material - black_material) / 39.0  # Normalize by total material

def create_opening_position():
    board = chess.Board()
    for _ in range(random.randint(5, 15)):
        moves = list(board.legal_moves)
        if moves:
            board.push(random.choice(moves))
        else:
            break
    return board

def main(test_mode=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ChessValueNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Hyperparameters
    batch_size = 64
    num_games = 100
    max_moves_per_game = 100 if test_mode else 200
    mcts_simulations = 50  # Increased from 50
    replay_buffer_size = 10000

    stats = {
        'games_played': 0,
        'total_losses': [],
        'avg_game_lengths': [],
    }

    if device.type == 'cuda':
        Thread(target=gpu_monitor, daemon=True).start()
        
    save_interval = 60 if test_mode else 600  # Seconds
    start_time = time.time()
    last_save_time = start_time

    print("Starting training...")
    
    replay_buffer = deque(maxlen=replay_buffer_size)

    for game_num in range(num_games):
        board = create_opening_position()
        game_moves = []

        for move_num in range(max_moves_per_game):
            if board.is_game_over():
                break

            root = Node(board)
            best_move = mcts_search(root, model, board_to_input, device, num_simulations=mcts_simulations)

            if best_move:
                board.push(best_move)
                game_moves.append(board.copy())
            else:
                print(f"No legal moves found in game {game_num + 1}, move {move_num + 1}")
                break

        evaluation = evaluate_game_result(board)
        
        # Add game positions to replay buffer
        for position in game_moves:
            replay_buffer.append((board_to_input(position, device), evaluate_game_result(position)))

        # Training step
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            batch_positions, batch_evals = zip(*batch)
            
            batch_positions = torch.stack(batch_positions).float()
            batch_evals = torch.tensor(batch_evals, device=device, dtype=torch.float32)

            optimizer.zero_grad()
            predicted_values = model(batch_positions).squeeze()
            loss = torch.nn.MSELoss()(predicted_values, batch_evals)
            loss.backward()
            optimizer.step()

            stats['total_losses'].append(loss.item())

        stats['games_played'] += 1
        stats['avg_game_lengths'].append(len(game_moves))

        print(f"Game {stats['games_played']}, Result: {evaluation:.2f}, Avg Loss: {loss.item():.4f}, Moves: {len(game_moves)}")

        if game_num % 10 == 0 and stats['total_losses']:
            avg_loss = sum(stats['total_losses'][-10:]) / min(len(stats['total_losses']), 10)
            scheduler.step(avg_loss)

        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            save_checkpoint(model, optimizer, stats, f"chess_model_checkpoint_{stats['games_played']}.pth")
            last_save_time = current_time

        elapsed_time = current_time - start_time
        print(f"Time elapsed: {elapsed_time:.2f}s")

        if test_mode and elapsed_time >= 300:
            print("Test mode time limit reached. Ending training.")
            break

    save_checkpoint(model, optimizer, stats, "chess_model_final.pth")
    with open("training_stats.json", "w") as f:
        json.dump(stats, f)

    print("Training complete. Model and stats saved.")
    print(f"Total games played: {stats['games_played']}")
    print(f"Average game length: {sum(stats['avg_game_lengths']) / len(stats['avg_game_lengths']):.2f} moves")
    if stats['total_losses']:
        print(f"Final average loss: {sum(stats['total_losses'][-1000:]) / min(len(stats['total_losses']), 1000):.4f}")
    else:
        print("No losses recorded.")

if __name__ == "__main__":
    main()