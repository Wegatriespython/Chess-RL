import os
import sys
import torch
import chess
import chess.engine
import json

def check_dependencies():
    print("Checking dependencies...")
    dependencies = [
        "torch", "chess", "numpy", "stockfish"
    ]
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} is installed")
        except ImportError:
            print(f"✗ {dep} is not installed. Please install it using 'pip install {dep}'")
            return False
    return True

def check_cuda():
    print("\nChecking CUDA availability...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("✗ CUDA is not available. Training will use CPU, which may be slow.")
        return False

def check_files():
    print("\nChecking necessary files...")
    files = [
        "main_training_script.py",
        "model.py",
        "mcts.py",
        "chess_utils.py"
        ]   
    all_files_present = True
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for file in files:
        file_path = os.path.join(script_dir, file)
        if os.path.isfile(file_path):
            print(f"✓ {file} is present")
        else:
            print(f"✗ {file} is missing")
            all_files_present = False
    return all_files_present

def check_stockfish():
    print("\nChecking Stockfish...")
    try:
        # Update this path to the correct location of your Stockfish executable
        stockfish_path = "C:\\Users\\vigne\\Downloads\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe"
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            print("✓ Stockfish is properly configured")
        return True
    except Exception as e:
        print(f"✗ Error with Stockfish: {str(e)}")
        print("Please ensure Stockfish is installed and the path is correctly specified in position_evaluation.py")
        return False

def test_model():
    print("\nTesting model initialization...")
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from model import ChessValueNetwork
        model = ChessValueNetwork()
        print("✓ Model initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Error initializing model: {str(e)}")
        return False

def test_minimal_training():
    print("\nRunning minimal training test...")
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from main_training_script import main
        # Redirect stdout to capture print statements
        original_stdout = sys.stdout
        sys.stdout = open('minimal_training_output.txt', 'w', encoding='utf-8')
        
        # Run main with a very small number of iterations
        main(test_mode=True)
        
        # Restore stdout
        sys.stdout.close()
        sys.stdout = original_stdout
        
        print("+ Minimal training completed successfully")
        print("Check 'minimal_training_output.txt' for details")
        return True
    except Exception as e:
        print(f"- Error during minimal training: {str(e)}")
        return False
def main():
    all_checks_passed = (
        check_dependencies() and
        check_cuda() and
        check_files() and
        check_stockfish() and
        test_model() and
        test_minimal_training()
    )

    if all_checks_passed:
        print("\nAll checks passed! You're ready to start training.")
    else:
        print("\nSome checks failed. Please address the issues before starting training.")

if __name__ == "__main__":
    main()