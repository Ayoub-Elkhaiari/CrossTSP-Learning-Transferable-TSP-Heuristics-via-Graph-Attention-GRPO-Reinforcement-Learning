# src/train.py
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
from src.environment import TSPEnv
from src.agent import GRPOAgent
from src.model import GATPolicy
import matplotlib.pyplot as plt
from src.evaluate import evaluate_tsp as evaluate_agent

def build_edge_index(n, device):
    """Fully connected graph without self-loops"""
    adj = torch.ones((n, n), dtype=torch.float32) - torch.eye(n)
    edge_index, _ = dense_to_sparse(adj)
    return edge_index.to(device)

def make_node_features(coords_tensor, visited_tensor, current_idx):
    """Add extra feature: distance to current node"""
    n = coords_tensor.shape[0]
    visited = visited_tensor.unsqueeze(1)
    current_mask = torch.zeros_like(visited)
    current_mask[current_idx] = 1.0
    # distance to current node
    dist_to_current = torch.norm(coords_tensor - coords_tensor[current_idx], dim=1, keepdim=True)
    x = torch.cat([coords_tensor, visited, current_mask, dist_to_current], dim=1)
    return x

def train_tsp(coords, num_episodes=2000, lr=1e-4, gamma=0.99, device="cpu", verbose=True):
    n = len(coords)
    env = TSPEnv(coords)
    model = GATPolicy(input_dim=5, hidden_dim=256, heads=8).to(device)
    agent = GRPOAgent(model, lr=lr, device=device)
    edge_index = build_edge_index(n, device)

    episode_rewards = []
    for episode in tqdm(range(num_episodes), desc="Training"):
        obs = env.reset()
        # random start city
        start = np.random.randint(n)
        env.current = start
        env.visited[start] = 1
        env.tour = [start]

        done = False
        log_probs = []
        rewards = []

        coords_tensor = torch.tensor(obs["coords"], dtype=torch.float32).to(device)
        visited = torch.tensor(obs["visited"], dtype=torch.float32).to(device)
        current = int(obs["current"])

        while not done:
            x = make_node_features(coords_tensor, visited, current)
            action, log_prob = agent.act(x, edge_index)
            obs, reward, done, _ = env.step(action)
            # reward shaping
            reward = reward / 100.0  # normalize distances
            if done:
                reward += 10.0  # bonus for completing tour
            rewards.append(reward)
            log_probs.append(log_prob.to(device))

            coords_tensor = torch.tensor(obs["coords"], dtype=torch.float32).to(device)
            visited = torch.tensor(obs["visited"], dtype=torch.float32).to(device)
            current = int(obs["current"])

        # compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # normalize
        if returns.std().item() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(log_probs)
        # entropy bonus
        probs = torch.softmax(model(make_node_features(coords_tensor, visited, current), edge_index), dim=-1)
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy().mean()
        loss = -torch.sum(log_probs * returns) - 0.01 * entropy
        agent.update(loss)

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        if verbose and (episode + 1) % 50 == 0:
            avg_r = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward (last 50): {avg_r:.3f}")

    print("Training complete!")
    return model, episode_rewards




# evaluation utilities retained (but call evaluate_tsp from here or separate script)
def evaluate_tsp(coords, model_path, device="cpu", visualize=False):
    n = len(coords)
    env = TSPEnv(coords)
    model = GATPolicy(input_dim=5, hidden_dim=256, heads=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    agent = GRPOAgent(model, device=device)

    # ✅ attach edge_index to agent (important fix)
    edge_index = build_edge_index(n, device)
    agent.edge_index = edge_index  # <--- this line fixes the AttributeError

    # ✅ call evaluator properly
    tour, metrics = evaluate_agent(agent, env, coords, L_opt=None, device=device, deterministic=True, use_2opt=True)

    L_pred = metrics["pred_length"]
    print(f"Evaluation Results - Predicted Tour Length: {L_pred:.2f}")

    if visualize:
        plt.figure(figsize=(8, 8))
        tour_coords = coords[tour]
        plt.plot(tour_coords[:, 0], tour_coords[:, 1], marker='o')
        for i, (x, y) in enumerate(tour_coords):
            plt.text(x, y, str(tour[i]), fontsize=12, color='red')
        plt.title(f"TSP Tour (Length: {L_pred:.2f})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

    return metrics
