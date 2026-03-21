# src/evaluate_dataset.py
import torch
import pandas as pd
import time
from src.utils import load_tsp_file
from src.environment import TSPEnv
from src.agent import GRPOAgent
from src.model import GATPolicy
from src.evaluate import evaluate_tsp
from torch_geometric.utils import dense_to_sparse

datasets = [
    ("data/berlin52.tsp", 7542, "gat_tsp_model.pt"),
    ("data/att48.tsp", 10628, "gat_tsp_model.pt"),
    ("data/eil51.tsp", 426, "gat_tsp_model.pt"),
    ("data/st70.tsp", 675, "gat_tsp_model.pt")
]

results = []
device = "cpu"

for path, L_opt, model_path in datasets:
    print(f"Evaluating {path} ...")
    coords = load_tsp_file(path, normalize=False)
    env = TSPEnv(coords)
    n = len(coords)

    model = GATPolicy(input_dim=5, hidden_dim=256, heads=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    agent = GRPOAgent(model, device=device)

    # Build fully connected edge_index and attach it to the agent
    adj = torch.ones((n, n), dtype=torch.float32) - torch.eye(n)
    edge_index, _ = dense_to_sparse(adj)
    agent.edge_index = edge_index.to(device)

    # ✅ measure execution time
    start_time = time.time()
    tour, metrics = evaluate_tsp(agent, env, coords, L_opt=L_opt, device=device, deterministic=True)
    end_time = time.time()
    exec_time = end_time - start_time
    metrics["exec_time(s)"] = exec_time

    print(f"Dataset: {path}")
    print(f"  Predicted length: {metrics['pred_length']:.2f}")
    if metrics['gap(%)'] is not None:
        print(f"  Optimal gap: {metrics['gap(%)']:.2f}%")
    if metrics['exec_time(s)'] is not None:
        print(f"  Execution time: {metrics['exec_time(s)']:.3f}s\n")
    else:
        print("  Execution time: N/A\n")

    results.append({
        "dataset": path.split("/")[-1],
        "pred_length": metrics["pred_length"],
        "optimal_length": L_opt,
        "gap(%)": metrics["gap(%)"],
        "exec_time(s)": metrics["exec_time(s)"]
    })

# Save metrics to CSV
df = pd.DataFrame(results)
df.to_csv("evaluation_metrics.csv", index=False)
print("✅ Saved evaluation metrics to evaluation_metrics.csv")
