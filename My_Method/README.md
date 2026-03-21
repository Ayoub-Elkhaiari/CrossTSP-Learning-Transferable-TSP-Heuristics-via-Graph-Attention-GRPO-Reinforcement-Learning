# TSP Solver with Graph Attention Networks and GRPO

A reinforcement learning approach to solving the Traveling Salesman Problem (TSP) using Graph Attention Networks (GAT) and Group Relative Policy Optimization (GRPO).

## Overview

This project implements a neural network-based solution for the TSP that learns to construct near-optimal tours through reinforcement learning. The model uses Graph Attention Networks to capture spatial relationships between cities and employs GRPO for training.

## Features

- **Graph Attention Network (GAT)** policy for learning node importance
- **GRPO-based training** with policy gradient optimization
- **Custom TSP environment** with Gym-like interface
- **Interactive visualization** of tour construction
- **Evaluation suite** for multiple benchmark datasets
- Support for standard TSPLIB format files

## Project Structure

```
candleofdiscovery-test/
├── evaluate_datasets.py    # Batch evaluation script
├── main.py                  # Main training/testing entry point
├── requirements.txt         # Python dependencies
├── data/                    # TSP benchmark datasets
│   ├── att48.tsp           # 48 US capitals
│   ├── berlin52.tsp        # 52 Berlin locations
│   ├── eil51.tsp           # 51-city Eilon problem
│   └── st70.tsp            # 70-city Smith problem
└── src/                     # Source code
    ├── agent.py            # GRPO agent implementation
    ├── environment.py      # TSP Gym environment
    ├── evaluate.py         # Evaluation utilities
    ├── model.py            # GAT policy network
    ├── train.py            # Training and evaluation loops
    └── utils.py            # Data loading utilities
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd candleofdiscovery-test
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `torch`
- `torch-geometric`
- `numpy`
- `pandas`
- `matplotlib`
- `gym`
- `tqdm`

## Usage

### Training

Train the model on a TSP dataset:

```bash
python main.py --data data/berlin52.tsp --episodes 500 --lr 1e-4 --device cpu
```

**Arguments:**
- `--data`: Path to TSP dataset (default: `data/berlin52.tsp`)
- `--episodes`: Number of training episodes (default: 500)
- `--lr`: Learning rate (default: 1e-4)
- `--device`: Device to use (`cpu` or `cuda`, default: `cpu`)

The trained model will be saved to `models/gat_tsp_model.pt`.

### Testing

Evaluate a trained model with visualization:

```bash
python main.py --data data/berlin52.tsp --test --device cpu
```

This will:
- Load the trained model from `models/gat_tsp_model.pt`
- Generate a tour using the learned policy
- Display an animated visualization of the tour construction
- Print the total tour distance

### Batch Evaluation

Evaluate the model on multiple benchmark datasets:

```bash
python evaluate_datasets.py
```

This script evaluates the model on all datasets in the `data/` directory and generates a CSV file with performance metrics including:
- Predicted tour length
- Optimal tour length (known from TSPLIB)
- Optimality gap (%)
- Execution time

Results are saved to `evaluation_metrics.csv`.

## Model Architecture

### GAT Policy Network

The policy network consists of:
- **Input layer**: 2D coordinates (x, y) for each city
- **GAT Layer 1**: Multi-head attention (4 heads) with hidden dimension 64
- **GAT Layer 2**: Single-head attention
- **Policy head**: Linear layer outputting action logits

### GRPO Agent

The agent uses:
- Policy gradient with discounted returns
- Masking mechanism to prevent revisiting cities
- Adam optimizer for policy updates
- Reward normalization for stable training

## TSP Environment

The custom Gym environment features:
- Fully connected graph representation
- Penalty for invalid moves (revisiting cities)
- Negative distance as step reward
- Optional real-time visualization

## Benchmark Datasets

The repository includes four TSPLIB benchmark problems:

| Dataset | Cities | Optimal Length | Description |
|---------|--------|----------------|-------------|
| att48 | 48 | 10,628 | 48 US state capitals |
| berlin52 | 52 | 7,542 | 52 locations in Berlin |
| eil51 | 51 | 426 | 51-city Eilon problem |
| st70 | 70 | 675 | 70-city Smith problem |

## Results

After training, the model learns to construct competitive tours. The evaluation script compares predicted tour lengths against known optimal solutions and reports the optimality gap.

## Visualization

The project includes two visualization modes:

1. **Training progress**: Plot of episode rewards over time
2. **Tour animation**: Real-time rendering of the agent constructing a tour, showing city-by-city decisions

## Future Improvements

- [ ] Implement attention visualization
- [ ] Add beam search for inference
- [ ] Support for larger problem instances
- [ ] Multi-agent training with experience replay
- [ ] Integration with OR-Tools for baseline comparison
- [ ] Hyperparameter tuning utilities

## References

- Graph Attention Networks (Veličković et al., 2018)
- Attention, Learn to Solve Routing Problems! (Kool et al., 2019)
- TSPLIB benchmark library

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
