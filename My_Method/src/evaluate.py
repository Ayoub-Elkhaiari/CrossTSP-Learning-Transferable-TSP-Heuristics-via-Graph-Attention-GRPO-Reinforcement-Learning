# src/evaluate.py
import numpy as np
import torch
from src.environment import TSPEnv

def compute_tour_length(coords, tour):
    total = 0.0
    n = len(tour)
    for i in range(n-1):
        total += np.linalg.norm(coords[tour[i]] - coords[tour[i+1]])
    return total

def two_opt(tour, coords):
    """Simple 2-opt local search to improve tour"""
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour)-2):
            for j in range(i+1, len(tour)-1):
                a, b = tour[i], tour[i+1]
                c, d = tour[j], tour[j+1]
                old_dist = np.linalg.norm(coords[a]-coords[b]) + np.linalg.norm(coords[c]-coords[d])
                new_dist = np.linalg.norm(coords[a]-coords[c]) + np.linalg.norm(coords[b]-coords[d])
                if new_dist < old_dist:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
    return tour

def evaluate_tsp(agent, env, coords_original, L_opt=None, device="cpu", deterministic=True, use_2opt=True):
    obs = env.reset()
    tour = [int(obs["current"])]
    done = False
    n = len(coords_original)
    coords_tensor = torch.tensor(coords_original, dtype=torch.float32).to(device)
    visited = torch.tensor(obs["visited"], dtype=torch.float32).to(device)
    current = int(obs["current"])

    # edge_index
    if hasattr(agent, "edge_index"):
        edge_index = agent.edge_index
    else:
        edge_index = None

    while not done:
        x = torch.cat([coords_tensor, visited.unsqueeze(1),
                       torch.eye(n, dtype=torch.float32)[current].unsqueeze(1).to(device),
                       torch.norm(coords_tensor - coords_tensor[current], dim=1, keepdim=True)], dim=1)
        with torch.no_grad():
            action, _ = agent.act(x, edge_index)
        obs, reward, done, _ = env.step(action)
        tour.append(int(action))
        visited = torch.tensor(obs["visited"], dtype=torch.float32).to(device)
        current = int(obs["current"])

    if tour[0] != tour[-1]:
        tour.append(tour[0])

    if use_2opt:
        tour = two_opt(tour, np.array(coords_original))

    L_pred = compute_tour_length(coords_original, tour)
    metrics = {
        "pred_length": L_pred,
        "optimal_length": L_opt,
        "gap(%)": ((L_pred - L_opt)/L_opt*100) if L_opt else None,
        "exec_time(s)": None
    }
    return tour, metrics
