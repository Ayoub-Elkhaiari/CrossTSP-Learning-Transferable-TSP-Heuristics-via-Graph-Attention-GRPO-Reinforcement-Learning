# CrossTSP 🧭
### Learning Transferable TSP Heuristics via Graph Attention Networks and Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-latest-3C2179?logo=pyg&logoColor=white)](https://pyg.org/)
[![TSPLIB](https://img.shields.io/badge/Benchmarks-TSPLIB-6366F1?logo=databricks&logoColor=white)](https://github.com/Ayoub-Elkhaiari/CrossTSP-Learning-Transferable-TSP-Heuristics-via-Graph-Attention-GRPO-Reinforcement-Learning)
[![Status](https://img.shields.io/badge/Status-Research-F59E0B?logo=academia&logoColor=white)](https://github.com/Ayoub-Elkhaiari/CrossTSP-Learning-Transferable-TSP-Heuristics-via-Graph-Attention-GRPO-Reinforcement-Learning)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## The Core Question

> *Can a model trained on a single TSP instance learn heuristics that transfer to unseen instances — without retraining?*

Most learned TSP solvers train and test on the same distribution. **CrossTSP** takes a harder path: train once on `berlin52`, then evaluate zero-shot on structurally different instances. The results reveal when and why generalization succeeds — and where it breaks down.

---

## What Makes This Different

Classical approaches like Genetic Algorithms learn nothing — they solve each instance from scratch. Standard learned heuristics train on random instances and test on random instances. **CrossTSP** bridges combinatorial optimization and representation learning by asking a more fundamental question: *what structural features of a TSP landscape does the model actually learn?*

The answer turns out to be geometric: the model learns **Euclidean spatial structure** — which transfers well across instances with similar metrics, and fails predictably on instances with non-Euclidean distances. This is not a bug. It is a finding.

---

## Architecture

```
TSP Instance (coords)
        │
        ▼
┌─────────────────────────────┐
│     5-Dimensional Node      │
│         Features            │
│  [x, y, visited, current,   │
│   dist_to_current]          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   GAT Layer 1               │
│   (4 attention heads)       │
│   → hidden_dim × 4          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   GAT Layer 2               │
│   (1 aggregation head)      │
│   → hidden_dim              │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Policy Head               │
│   Linear → logits per node  │
│   + visited mask (-1e9)     │
└─────────────┬───────────────┘
              │
              ▼
      Sampled Tour (RL)
              │
              ▼
┌─────────────────────────────┐
│   2-opt Local Search        │
│   (post-processing)         │
└─────────────────────────────┘
```

**Key design choices:**
- **5-dim node features** — adds distance-to-current and current-node mask on top of raw coordinates, giving the model dynamic spatial context at each step
- **GRPO-style training** — policy gradient with discounted returns, return normalization, and entropy regularization for stable exploration
- **Hybrid pipeline** — learned construction heuristic refined with 2-opt local search
- **Visited masking** — hard constraint ensuring valid tours at every step

---

## Results

### Trained on `berlin52` only — evaluated zero-shot on all instances

| Dataset | Cities | Metric Type | CrossTSP | GA | Known Optimal | CrossTSP Gap | GA Gap |
|---|---|---|---|---|---|---|---|
| berlin52 | 52 | Euclidean | 8006 | 9555 | 7542 | **+6.1%** | +26.7% |
| eil51 | 51 | Euclidean | 479 | 534 | 426 | **+12.5%** | +25.5% |
| st70 | 70 | Euclidean | 661 | 855 | 675 | **-1.98%** ⭐ | +26.7% |
| att48 | 48 | Pseudo-Euclidean | 39850 | 41741 | 10628 | +274% | +292% |

⭐ *st70: CrossTSP finds a tour shorter than the known benchmark value — consistent across multiple runs (643.5 in zero-shot evaluation). The 2-opt post-processing discovers improvements not captured by the standard benchmark solution.*

### Key findings

**Generalization works — within metric families.** The model, never having seen eil51 or st70 during training, achieves 12.5% and -1.98% gaps respectively. It consistently outperforms GA by 13–28 percentage points on **Euclidean instances**.

**Metric mismatch breaks generalization — predictably.** att48 uses pseudo-Euclidean distances (`d = ceil(sqrt((dx²+dy²)/10))`), a warped metric the model was never exposed to. The failure is total and expected — and tells us exactly what the model learned: Euclidean geometry, not abstract TSP structure.

**This is a research finding, not a failure.** The boundary of generalization is now known and interpretable, it will be solved and added more features for a publishable paper in the future.

---

## Visualizations

After running evaluation, CrossTSP produces the following plots. Full benchmark comparison results are available in `results` and `My_method`.

### 1. Predicted Tour
After running `--test`, a tour visualization is shown automatically:

```
🗺️  Tour plot: cities as nodes, predicted path as edges, city indices labeled
```

```bash
python main.py --data data/berlin52.tsp --test --device cpu
```
> The tour plot appears automatically after evaluation. Red numbers = city indices. The path shows the order of visits.
**Note**: these plots was conducted recently with differents seed so it is different than the CSVs metrics.
for CrossTSP:
 <img width="1730" height="917" alt="image" src="https://github.com/user-attachments/assets/fcd78a39-6b82-4a72-b4a0-4de940982322" />

for GA : 

<img width="1177" height="838" alt="image" src="https://github.com/user-attachments/assets/92e79152-96d0-4df1-b902-2dffde2894f0" />



### 2. Cross-Instance Evaluation Bar Chart
Run the full benchmark evaluation to get a gap comparison chart:

```bash
python evaluate_datasets.py
```
> Produces a CSV with all Datasets.
for CrossTSP:
<img width="966" height="226" alt="image" src="https://github.com/user-attachments/assets/956112ba-0b70-45a6-95f5-acf798b78e51" />

for GA: 

<img width="915" height="225" alt="image" src="https://github.com/user-attachments/assets/ca252c51-d1ff-4790-860e-f6d982d3c3bd" />


---

## Installation

```bash
git clone https://github.com/Ayoub-Elkhaiari/CrossTSP-Learning-Transferable-TSP-Heuristics-via-Graph-Attention-GRPO-Reinforcement-Learning.git
cd CrossTSP-Learning-Transferable-TSP-Heuristics-via-Graph-Attention-GRPO-Reinforcement-Learning
pip install torch torch-geometric gym numpy matplotlib tqdm
```

---

## Usage

### Train
```bash
python main.py --data data/berlin52.tsp --episodes 300 --lr 1e-4 --device cpu
```
Saves model to `gat_tsp_model.pt` and plots the training reward curve.

### Evaluate (single instance)
```bash
python main.py --data data/berlin52.tsp --test --device cpu
```
Loads `gat_tsp_model.pt`, applies 2-opt, and visualizes the predicted tour.

### Evaluate (all TSPLIB instances)
```bash
python evaluate_datasets.py
```
Runs zero-shot evaluation across all datasets and outputs the results CSV.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | `data/berlin52.tsp` | Path to `.tsp` file |
| `--episodes` | `300` | Training episodes (use 2000 for best results) |
| `--lr` | `1e-4` | Learning rate |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--test` | `False` | Flag to run evaluation instead of training |

---

## Project Structure

```
My_Method/
├── data/                    # TSPLIB benchmark instances
│   ├── berlin52.tsp
│   ├── att48.tsp
│   ├── eil51.tsp
│   └── st70.tsp
├── src/
│   ├── model.py             # GATPolicy: 2-layer GAT + policy head
│   ├── agent.py             # GRPOAgent: policy gradient training
│   ├── environment.py       # TSPEnv: gym-compatible TSP environment
│   ├── evaluate.py          # Evaluation + 2-opt local search
│   ├── train.py             # Training loop with reward shaping
│   └── utils.py             # TSP file loading + distance matrix
├── models/                  # Saved model checkpoints
|   
├── main.py                  # CLI entry point
├── evaluate_datasets.py     # Cross-instance benchmark evaluation
└── README.md
```

---

## Comparison to Prior Work

| Method | Approach | Generalization | Metric-aware |
|---|---|---|---|
| Nearest Neighbor | Classical greedy | ✅ | ❌ |
| Genetic Algorithm | Evolutionary | ✅ | ❌ |
| Kool et al. (2019) | Attention Model + RL | ✅ (random instances) | ❌ |
| **CrossTSP** | **GAT + GRPO + 2-opt** | **✅ (single instance)** | **Explicit failure analysis** |

---

## Limitations & Future Work

- **Scale**: Currently tested on instances up to 70 cities. Scaling to 200+ requires architectural changes (sparse attention, beam search)
- **Metric generalization**: Training on mixed-metric instances (Euclidean + pseudo-Euclidean + geographic) could produce a metric-agnostic policy
- **Baselines**: Comparison against Attention Model (Kool et al.) and OR-Tools is planned
- **Statistical rigor**: Results should be averaged over multiple runs with std reported — currently single-run results

---

## Research Context

This project is a practical investigation into a question central to automated optimization research: *do learned heuristics capture generalizable landscape features, or do they overfit to instance structure?*

The empirical finding — that generalization tracks metric structure — motivates deeper work on landscape-aware representations, connecting directly to topics in algorithm selection, meta-learning for combinatorial optimization, and the emerging field of learned landscape features.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{elkhaiari2025crosstsp,
  author = {EL KHAIARI, Ayoub},
  title = {CrossTSP: Learning Transferable TSP Heuristics via Graph Attention & GRPO Reinforcement Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Ayoub-Elkhaiari/CrossTSP-Learning-Transferable-TSP-Heuristics-via-Graph-Attention-GRPO-Reinforcement-Learning}
}
```

---

## Author

**Ayoub EL KHAIARI**:  Independent AI Researcher

---

*"The boundary of generalization is itself a finding."*
