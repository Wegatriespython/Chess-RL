import torch
from model import ChessValueNetwork

def save_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessValueNetwork().to(device)
    
    # Here, we assume that the model's parameters are still in memory
    # If they're not, you'll need to retrain the model or load it from somewhere

    # Save the model
    torch.save(model.state_dict(), "chess_model.pth")
    print("Model saved as chess_model.pth")

if __name__ == "__main__":
    save_model()