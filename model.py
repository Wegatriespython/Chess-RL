import torch
import torch.nn as nn

class ChessValueNetwork(nn.Module):
    def __init__(self):
        super(ChessValueNetwork, self).__init__()
        self.fc1 = nn.Linear(64 * 12, 256)  # 12 piece types * 64 squares
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)  # Remove the last dimension if it's 1