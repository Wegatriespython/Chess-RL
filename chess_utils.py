import torch

def board_to_input(board, device):
    pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    board_state = torch.zeros(12, 8, 8, device=device)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            idx = pieces.index(piece.symbol())
            board_state[idx, i // 8, i % 8] = 1
    return board_state.flatten()