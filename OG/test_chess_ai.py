import chess
import torch
from model import ChessValueNetwork
from chess_utils import board_to_input
from mcts import Node, mcts_search

def play_game(model, device, player_color=chess.WHITE):
    board = chess.Board()
    
    while not board.is_game_over():
        if board.turn == player_color:
            # Player's turn
            print(board)
            while True:
                move = input("Enter your move (e.g., e2e4): ")
                try:
                    board.push_san(move)
                    break
                except ValueError:
                    print("Invalid move. Try again.")
        else:
            # AI's turn
            root = Node(board)
            best_move = mcts_search(root, model, board_to_input, device, num_simulations=100)
            if best_move:
                print(f"AI plays: {best_move}")
                board.push(best_move)
            else:
                print("AI couldn't find a valid move.")
                break

    print(board)
    print("Game Over")
    print(f"Result: {board.result()}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessValueNetwork().to(device)
    
    # Load the trained model
    model.load_state_dict(torch.load("chess_model.pth"))
    model.eval()

    player_color = input("Do you want to play as White or Black? (W/B): ").upper()
    player_color = chess.WHITE if player_color == 'W' else chess.BLACK

    play_game(model, device, player_color)

if __name__ == "__main__":
    main()