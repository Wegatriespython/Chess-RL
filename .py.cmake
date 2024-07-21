import torch
import sys

def check_cuda():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    print("\nCUDA availability:")
    print(f"CUDA is available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nCUDA Device {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    print("\nDefault device:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    print("\nTesting CUDA:")
    try:
        x = torch.rand(5, 3)
        print(f"Random tensor on CPU:\n{x}")
        if torch.cuda.is_available():
            x = x.cuda()
            print(f"Same tensor on GPU:\n{x}")
        print("CUDA test successful!")
    except Exception as e:
        print(f"CUDA test failed. Error: {e}")

if __name__ == "__main__":
    check_cuda()