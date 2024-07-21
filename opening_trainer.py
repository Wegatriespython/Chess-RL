import chess
import torch
import time
import random
from model import ChessValueNetwork
from mcts import Node, mcts_search
from chess_utils import board_to_input
from position_evaluation import evaluate_position

def is_opening_complete(board):
    # Improved criteria for opening completion
    return board.fullmove_number > 15 and \
           board.castling_rights == chess.BB_EMPTY and \
           sum(1 for piece in board.piece_map().values() if piece.piece_type in [chess.KNIGHT, chess.BISHOP]) >= 6 and \
           len(board.attackers(chess.WHITE, chess.E4 | chess.D4 | chess.E5 | chess.D5)) >= 2 and \
           len(board.attackers(chess.BLACK, chess.E4 | chess.D4 | chess.E5 | chess.D5)) >= 2

def generate_opening_position():
    board = chess.Board()
    moves_played = 0
    while moves_played < 15 or not is_opening_complete(board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        board.push(move)
        moves_played += 1
    return board

def train_opening(model, optimizer, device, num_games=100, time_limit=60):
    losses = []
    start_time = time.time()

    for game in range(num_games):
        if time.time() - start_time > time_limit:
            break

        board = generate_opening_position()
        game_moves = []  # Store (board_state, move) pairs

        while not is_opening_complete(board):
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

            # (Optional) Keep track of losses if you want to analyze them
            losses.append(loss.item())

        if game % 10 == 0:
            print(f"Opening game {game}, Loss: {loss.item()}")

    return model, losses