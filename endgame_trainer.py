import chess
import torch
import time
import random
from model import ChessValueNetwork
from mcts import Node, mcts_search
from chess_utils import board_to_input
from position_evaluation import evaluate_position

def is_endgame(board):
    # Define criteria for when endgame begins
    # This could be based on material count, queen exchanges, etc.
    return len(board.piece_map()) <= 12 or \
           not any(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))

def generate_endgame_position():
    board = chess.Board()
    # Simulate a game until we reach an endgame position
    while not is_endgame(board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        board.push(move)
    return board

def train_endgame(model, optimizer, device, num_games=100, time_limit=60):
    losses = []
    start_time = time.time()

    for game in range(num_games):
        if time.time() - start_time > time_limit:
            break

        board = generate_endgame_position()
        game_moves = []  # Store (board_state, move) pairs

        while not board.is_game_over():  # Play until checkmate, stalemate, etc.
            root = Node(board)
            best_move = mcts_search(root, model, board_to_input, device)
            if best_move:
                board.push(best_move)
                game_moves.append((board.copy(), best_move))
            else:
                break

        # Evaluate the final position ONLY ONCE
        evaluation = evaluate_position(board)

        # Update model for EACH MOVE in the game using the final evaluation
        for board_state, move in game_moves:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                board_input = board_to_input(board_state, device).unsqueeze(0)
                predicted_value = model(board_input).view(-1)
                target = torch.tensor([evaluation], device=device, dtype=torch.float32).view(-1)
                loss = torch.nn.MSELoss()(predicted_value, target)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())  # (Optional)

        if game % 10 == 0:
            print(f"Endgame game {game}, Loss: {loss.item()}")

    return model, losses