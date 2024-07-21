import torch
import torch.optim as optim
from model import ChessValueNetwork
from chess_utils import board_to_input
from train import train

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ChessValueNetwork().to(device)
    optimizer = optim.Adam(model.parameters())
    train(model, optimizer, board_to_input, device)

    # Save the model after training
    torch.save(model.state_dict(), "chess_model.pth")
    print("Model saved as chess_model.pth")

if __name__ == "__main__":
    main()