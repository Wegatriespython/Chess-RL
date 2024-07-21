import time
import torch
import chess
import torch.nn as nn
from mcts import Node, mcts_search

def train_with_timing(model, optimizer, board_to_input, device, num_episodes=10):
    total_time = 0
    for episode in range(num_episodes):
        start_time = time.time()
        
        board = chess.Board()
        while not board.is_game_over():
            root = Node(board)
            best_move = mcts_search(root, model, board_to_input, device)
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
        value = model(board_to_input(board, device).unsqueeze(0))
        loss = nn.MSELoss()(value, torch.tensor([[reward]], dtype=torch.float32, device=device))
        loss.backward()
        optimizer.step()
        
        episode_time = time.time() - start_time
        total_time += episode_time
        
        if episode % 10 == 0:
            avg_time = total_time / (episode + 1)
            estimated_total = avg_time * num_episodes
            print(f"Episode {episode}, Loss: {loss.item()}, Time: {episode_time:.2f}s, Avg Time: {avg_time:.2f}s")
            print(f"Estimated total time: {estimated_total/3600:.2f} hours")

    print(f"Training completed. Total time: {total_time/3600:.2f} hours")

