import math
import random
import chess
import torch

class Node:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0

    def uct_value(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)

def mcts_search(root, model, board_to_input, device, num_simulations=100):
    for _ in range(num_simulations):
        node = root
        # Selection
        while node.children and not node.board.is_game_over():
            node = max(node.children, key=lambda n: n.uct_value())
        
        # Expansion
        if not node.board.is_game_over():
            legal_moves = list(node.board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
                new_board = node.board.copy()
                new_board.push(move)
                new_node = Node(new_board, parent=node, move=move)
                node.children.append(new_node)
                node = new_node
        
        # Evaluation
        with torch.no_grad():
            board_input = board_to_input(node.board, device).unsqueeze(0)
            value = model(board_input).item()
        
        # Backpropagation
        while node:
            node.visits += 1
            node.value += value if node.board.turn == chess.WHITE else -value
            node = node.parent
    
    # Select best move
    if root.children:
        best_child = max(root.children, key=lambda n: n.visits)
        return best_child.move
    return None

def get_move_probabilities(root):
    total_visits = sum(child.visits for child in root.children)
    return {child.move: child.visits / total_visits for child in root.children}