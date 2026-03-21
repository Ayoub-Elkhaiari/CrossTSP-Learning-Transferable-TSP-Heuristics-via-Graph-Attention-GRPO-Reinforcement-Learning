import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import os

np.random.seed(42)
random.seed(42)

# --- Load TSP dataset (TSPLIB format) ---
def load_tsp(path):
    coords = []
    with open(path) as f:
        lines = f.readlines()
        start = False
        for line in lines:
            if "NODE_COORD_SECTION" in line:
                start = True
                continue
            if "EOF" in line:
                break
            if start:
                parts = line.split()
                coords.append((float(parts[1]), float(parts[2])))
    return np.array(coords)

# --- Utility functions ---
def distance(a, b):
    return np.linalg.norm(a - b)

def total_distance(tour, coords):
    return sum(distance(coords[tour[i]], coords[tour[(i+1)%len(tour)]]) for i in range(len(tour)))

# --- GA Core ---
def initialize_population(n_cities, pop_size):
    return [random.sample(range(n_cities), n_cities) for _ in range(pop_size)]

def tournament_selection(pop, fitness, k=5):
    selected = random.sample(list(zip(pop, fitness)), k)
    return min(selected, key=lambda x: x[1])[0]

def order_crossover(parent1, parent2):
    a, b = sorted(random.sample(range(len(parent1)), 2))
    child = [None]*len(parent1)
    child[a:b] = parent1[a:b]
    ptr = 0
    for city in parent2:
        if city not in child:
            while child[ptr] is not None:
                ptr += 1
            child[ptr] = city
    return child

def swap_mutation(tour):
    a, b = random.sample(range(len(tour)), 2)
    tour[a], tour[b] = tour[b], tour[a]
    return tour

# --- Visualization helper ---
def plot_tour(coords, tour, title, filename):
    ordered = coords[tour + [tour[0]]]
    plt.figure(figsize=(7, 5))
    plt.plot(ordered[:,0], ordered[:,1], '-o', markersize=5)
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i+1), fontsize=7, color="darkblue")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_convergence(history, title, filename):
    plt.figure(figsize=(6,4))
    plt.plot(history, label="Best Distance")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# --- GA with tracking ---
def genetic_algorithm(coords, pop_size=200, generations=300):
    pop = initialize_population(len(coords), pop_size)
    best = None
    best_fit = math.inf
    best_history = []

    for gen in range(generations):
        fitness = [total_distance(t, coords) for t in pop]
        new_pop = []
        elite = min(pop, key=lambda t: total_distance(t, coords))
        new_pop.append(elite)

        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fitness)
            p2 = tournament_selection(pop, fitness)
            child = order_crossover(p1, p2)
            if random.random() < 0.1:
                child = swap_mutation(child)
            new_pop.append(child)

        pop = new_pop
        current_best = min(fitness)
        best_history.append(current_best)

        if current_best < best_fit:
            best_fit = current_best
            best = pop[np.argmin(fitness)]

        if gen % 50 == 0:
            print(f"Generation {gen}, best distance: {best_fit:.2f}")

    return best, best_fit, best_history


# --- Run for all datasets ---
datasets = {
    "berlin52": {"path": "berlin52.tsp", "optimal": 7542},
    "att48": {"path": "att48.tsp", "optimal": 10628},
    "eil51": {"path": "eil51.tsp", "optimal": 426},
    "st70": {"path": "st70.tsp", "optimal": 675},
}

results = []
os.makedirs("results", exist_ok=True)

for name, info in datasets.items():
    print(f"\n--- Running GA on {name.upper()} ---")
    coords = load_tsp(info["path"])
    best_tour, best_dist, history = genetic_algorithm(coords, generations=300)

    gap = ((best_dist - info["optimal"]) / info["optimal"]) * 100
    results.append({
        "Dataset": name,
        "#Cities": len(coords),
        "Known Optimal": info["optimal"],
        "GA Best Distance": round(best_dist, 2),
        "GAP (%)": round(gap, 2)
    })

    # Save visualizations
    plot_tour(coords, best_tour, f"{name.upper()} Best Tour (Dist={best_dist:.2f})", f"results/best_tour_{name}.png")
    plot_convergence(history, f"GA Convergence on {name.upper()}", f"results/convergence_{name}.png")

# --- Save CSV summary ---
df = pd.DataFrame(results)
df.to_csv("results/GA_TSP_summary.csv", index=False)
print("\n✅ Results saved to 'results/GA_TSP_summary.csv'")
print(df)
