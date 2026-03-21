import argparse
import torch
import matplotlib.pyplot as plt
from src.utils import load_tsp_file, load_tsp_file_without_normalization
from src.train import train_tsp, evaluate_tsp

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards over Episodes")
    plt.grid(True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate GAT+GRPO on a TSP instance")
    parser.add_argument("--data", type=str, default="data/berlin52.tsp", help="Path to TSP dataset")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--test", action="store_true", help="Evaluate the trained model instead of training")
    args = parser.parse_args()

    coords = load_tsp_file(args.data)
    print(f"Loaded dataset with {len(coords)} cities from {args.data}")
    coords_ = load_tsp_file_without_normalization(args.data)
    if args.test:
        evaluate_tsp(coords_, model_path="gat_tsp_model.pt", device=args.device, visualize=True)
    else:
        model, rewards = train_tsp(coords, num_episodes=args.episodes, lr=args.lr, device=args.device)
        torch.save(model.state_dict(), "gat_tsp_model.pt")
        print("Model saved to gat_tsp_model.pt")
        plot_rewards(rewards)


if __name__ == "__main__":
    main()