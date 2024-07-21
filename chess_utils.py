import chess
import torch

def board_to_input(board, device):
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    colors = [chess.WHITE, chess.BLACK]
    
    input_tensor = torch.zeros(12, 8, 8, device=device)
    
    for color_idx, color in enumerate(colors):
        for piece_idx, piece_type in enumerate(piece_types):
            for square in board.pieces(piece_type, color):
                rank, file = divmod(square, 8)
                input_tensor[color_idx * 6 + piece_idx][rank][file] = 1
    
    return input_tensor

def mask_illegal_moves(board, move_probabilities):
    legal_moves = set(board.legal_moves)
    return {move: prob for move, prob in move_probabilities.items() if move in legal_moves}