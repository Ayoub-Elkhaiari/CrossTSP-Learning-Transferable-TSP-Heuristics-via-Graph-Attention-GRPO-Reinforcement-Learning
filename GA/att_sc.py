import numpy as np
import random
import math
import matplotlib.pyplot as plt

# --- Load ATT48 dataset ---
def load_tsp(path="att48.tsp"):
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

# --- ATT distance (pseudo-Euclidean) ---
def att_distance(a, b):
    rij = math.sqrt(((a[0]-b[0])**2 + (a[1]-b[1])**2) / 10.0)
    tij = int(rij + 0.5)
    return tij

def total_distance(tour, coords):
    return sum(att_distance(coords[tour[i]], coords[tour[(i+1)%len(tour)]]) for i in range(len(tour)))

# --- GA operators ---
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

# --- Visualization ---
def plot_tour(coords, tour, title="ATT48 Best Tour", save_path="att48_best_tour.png"):
    ordered = coords[tour + [tour[0]]]
    plt.figure(figsize=(7, 5))
    plt.plot(ordered[:,0], ordered[:,1], '-o', markersize=4)
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i+1), fontsize=6)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# --- Genetic Algorithm ---
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

# --- Run for ATT48 ---
if __name__ == "__main__":
    coords = load_tsp("att48.tsp")
    optimal_distance = 10628

    best_tour, best_dist, history = genetic_algorithm(coords, generations=300)

    print("\nFinal best distance:", best_dist)
    gap = ((best_dist - optimal_distance) / optimal_distance) * 100
    print(f"GAP vs Optimal: {gap:.2f}%")

    # Save visualization
    plot_tour(coords, best_tour, title=f"ATT48 GA Best Tour ({best_dist:.2f})")

    # Convergence curve
    plt.figure(figsize=(6,4))
    plt.plot(history, label="Best Distance")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.title("GA Convergence Curve on ATT48 (ATT Distance)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("att48_convergence.png", dpi=300)
    plt.close()

    print("Saved: att48_best_tour.png and att48_convergence.png")
