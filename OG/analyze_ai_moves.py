import chess
import chess.engine
import torch
from model import ChessValueNetwork
from chess_utils import board_to_input
from mcts import Node, mcts_search

def analyze_game(model, device, num_moves=20):
    board = chess.Board()
    # Replace this path with the actual path to your Stockfish executable
    stockfish_path = r"C:\Users\vigne\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"  # Use 'r' prefix for Windows paths
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    for _ in range(num_moves):
        if board.is_game_over():
            break

        root = Node(board)
        best_move = mcts_search(root, model, board_to_input, device, num_simulations=100)
        
        if best_move:
            # Analyze the move
            info = engine.analyse(board, chess.engine.Limit(time=0.1), root_moves=[best_move])
            score = info["score"].relative.score(mate_score=10000)
            
            print(f"Position: {board.fen()}")
            print(f"AI's move: {best_move}")
            print(f"Stockfish evaluation: {score/100:.2f}")  # Convert centipawns to pawns
            
            board.push(best_move)
        else:
            print("AI couldn't find a valid move.")
            break

    engine.quit()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessValueNetwork().to(device)
    
    # Load the trained model
    model.load_state_dict(torch.load("chess_model.pth"))
    model.eval()

    analyze_game(model, device)

if __name__ == "__main__":
    main()