import numpy as np
import chess
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network for the value function
class ChessValueNetwork(nn.Module):
    def __init__(self):
        super(ChessValueNetwork, self).__init__()
        self.fc1 = nn.Linear(64 * 12, 256)  # 12 piece types * 64 squares
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# Convert chess board to input tensor
def board_to_input(board):
    pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    board_state = torch.zeros(12, 8, 8)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            idx = pieces.index(piece.symbol())
            board_state[idx, i // 8, i % 8] = 1
    return board_state.flatten()

# MCTS node
class Node:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0

    def uct_value(self, c_param=1.41):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c_param * np.sqrt(np.log(self.parent.visits) / self.visits) if self.parent else 0

# MCTS search function
def mcts_search(root, model, num_simulations=100):
    for _ in range(num_simulations):
        node = root
        # Selection
        while node.children and not node.board.is_game_over():
            node = max(node.children, key=lambda n: n.uct_value())
        
        # Expansion
        if not node.board.is_game_over():
            legal_moves = list(node.board.legal_moves)
            if legal_moves:
                move = np.random.choice(legal_moves)
                new_board = node.board.copy()
                new_board.push(move)
                new_node = Node(new_board, parent=node, move=move)
                node.children.append(new_node)
                node = new_node
        
        # Simulation (using the value network instead of random rollout)
        with torch.no_grad():
            value = model(board_to_input(node.board).unsqueeze(0)).item()
        
        # Backpropagation
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
    
    return max(root.children, key=lambda n: n.visits).move if root.children else None

# Training loop
def train(model, optimizer, num_episodes=1000):
    for episode in range(num_episodes):
        board = chess.Board()
        while not board.is_game_over():
            root = Node(board)
            best_move = mcts_search(root, model)
            if best_move is None:
                break
            board.push(best_move)
        
        # Get game result
        result = board.result()
        if result == '1-0':
            reward = 1
        elif result == '0-1':
            reward = -1
        else:
            reward = 0
        
        # Update model
        optimizer.zero_grad()
        value = model(board_to_input(board).unsqueeze(0))
        loss = nn.MSELoss()(value, torch.tensor([[reward]], dtype=torch.float32))
        loss.backward()
        optimizer.step()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {loss.item()}")

# Main function
def main():
    model = ChessValueNetwork()
    optimizer = optim.Adam(model.parameters())
    train(model, optimizer)

if __name__ == "__main__":
    main()