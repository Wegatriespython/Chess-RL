import torch
import time
import json
from threading import Thread
from model import ChessValueNetwork
from opening_trainer import train_opening, generate_opening_position
from midgame_trainer import train_midgame, generate_midgame_position
from endgame_trainer import train_endgame, generate_endgame_position

def save_checkpoint(model, optimizer, stats, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, filename)

def gpu_monitor():
    while True:
        gpu = torch.cuda.get_device_properties(0)
        utilization = torch.cuda.utilization(0)
        print(f"GPU Utilization: {utilization}%, Memory: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB / {gpu.total_memory/1024**3:.2f}GB")
        time.sleep(5)  # Update every 5 seconds

def main(test_mode=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ChessValueNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

 # Hyperparameters
    batch_size = 8 if test_mode else 8  # Reduced batch size for test mode
    num_batches = 2 if test_mode else 10  # Reduced number of batches for test mode

    # Training parameters
    time_limit = 30 if test_mode else 3600  # 30 seconds for test, 1 hour for actual training
    save_interval = 15 if test_mode else 600  # 15 seconds for test, 10 minutes for actual training
    phase_time_limit = 5 if test_mode else 60  #

    stats = {
        'opening_games': 0,
        'midgame_positions': 0,
        'endgame_positions': 0,
        'total_losses': []
    }

    # Start GPU monitoring
    Thread(target=gpu_monitor, daemon=True).start()

    start_time = time.time()
    last_save_time = start_time

    print("Starting training...")
    while time.time() - start_time < time_limit:
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Train opening
        print("Training opening...")
        model, opening_losses = train_opening(model, optimizer, device, num_games=1 if test_mode else 100, time_limit=phase_time_limit)
        stats['opening_games'] += 1 if test_mode else 100
        stats['total_losses'].extend(opening_losses)

        # Train midgame
        print("Training midgame...")
        model, midgame_losses = train_midgame(model, optimizer, device, num_positions=1 if test_mode else 200, time_limit=phase_time_limit)
        stats['midgame_positions'] += 1 if test_mode else 200
        stats['total_losses'].extend(midgame_losses)

        # Train endgame
        print("Training endgame...")
        model, endgame_losses = train_endgame(model, optimizer, device, num_positions=1 if test_mode else 200, time_limit=phase_time_limit)
        stats['endgame_positions'] += 1 if test_mode else 200
        stats['total_losses'].extend(endgame_losses)


        # Save checkpoint if save_interval has passed
        if current_time - last_save_time >= save_interval:
            save_checkpoint(model, optimizer, stats, f"chess_model_checkpoint_{int(elapsed_time)}s.pth")
            last_save_time = current_time

        print(f"Time elapsed: {elapsed_time:.2f}s")

    # Save final model and stats
    save_checkpoint(model, optimizer, stats, "chess_model_final.pth")
    
    # Save stats as JSON for easy analysis
    with open("training_stats.json", "w") as f:
        json.dump(stats, f)

    print("Training complete. Model and stats saved.")

if __name__ == "__main__":
    main()