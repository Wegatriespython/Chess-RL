import chess
import chess.engine
import torch
import time
from model import ChessValueNetwork
from chess_utils import board_to_input
from mcts import Node, mcts_search

def test_scenario(model, device, fen, num_moves=5):
    board = chess.Board(fen)
    total_time = 0
    
    for _ in range(num_moves):
        if board.is_game_over():
            break
        
        start_time = time.time()
        root = Node(board)
        best_move = mcts_search(root, model, board_to_input, device, num_simulations=100)
        end_time = time.time()
        
        move_time = end_time - start_time
        total_time += move_time
        
        if best_move:
            print(f"Position: {board.fen()}")
            print(f"AI's move: {best_move}")
            print(f"Time taken: {move_time:.2f} seconds")
            board.push(best_move)
        else:
            print("AI couldn't find a valid move.")
            break
    
    print(f"Average move time: {total_time/num_moves:.2f} seconds")

def self_play(model, device, num_games=10):
    wins = {'white': 0, 'black': 0, 'draw': 0}
    
    for _ in range(num_games):
        board = chess.Board()
        while not board.is_game_over():
            root = Node(board)
            best_move = mcts_search(root, model, board_to_input, device, num_simulations=100)
            if best_move:
                board.push(best_move)
            else:
                break
        
        result = board.result()
        if result == '1-0':
            wins['white'] += 1
        elif result == '0-1':
            wins['black'] += 1
        else:
            wins['draw'] += 1
    
    print(f"Self-play results: {wins}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessValueNetwork().to(device)
    
    # Load the trained model
    model.load_state_dict(torch.load("chess_model.pth"))
    model.eval()

    # Test specific scenarios
    scenarios = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # After 1. e4 e5 2. Nf3 Nc6
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3",  # After 1. e4 e5 2. Nc3 Nf6
    ]

    for scenario in scenarios:
        print(f"Testing scenario: {scenario}")
        test_scenario(model, device, scenario)
        print()

    # Self-play test
    self_play(model, device)

if __name__ == "__main__":
    main()