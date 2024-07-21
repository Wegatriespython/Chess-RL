import chess
import chess.engine

def evaluate_position(board, engine_depth=10):
    # Using double backslashes
    stockfish_path = "C:\\Users\\vigne\\Downloads\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe"
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path, set_options={"Threads": 4})
    result = engine.analyse(board, chess.engine.Limit(depth=engine_depth))
    score = result["score"].relative.score()
    
    if score is not None:
        # Normalize the score to be between -1 and 1
        return max(min(score / 1000, 1), -1)
    else:
        # If score is None, it's likely a forced mate
        return 1 if result["score"].relative.is_mate() else -1

# Example usage:
# board = chess.Board()
# evaluation = evaluate_position(board)
# print(f"Position evaluation: {evaluation}")