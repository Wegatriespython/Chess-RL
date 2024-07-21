import numpy as np
import torch

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

def mcts_search(root, model, board_to_input, device, num_simulations=50):
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
            value = model(board_to_input(node.board, device).unsqueeze(0)).item()
        
        # Backpropagation
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
    
    return max(root.children, key=lambda n: n.visits).move if root.children else None