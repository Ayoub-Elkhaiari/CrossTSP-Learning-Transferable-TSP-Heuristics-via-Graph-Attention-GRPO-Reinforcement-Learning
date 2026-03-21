import numpy as np

def load_tsp_file_without_normalization(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    coords = []
    start = False
    for line in lines:
        if "NODE_COORD_SECTION" in line:
            start = True
            continue
        if start:
            if "EOF" in line:
                break
            parts = line.strip().split()
            if len(parts) >= 3:
                coords.append([float(parts[1]), float(parts[2])])
    return np.array(coords)

# import numpy as np

def load_tsp_file(filepath, normalize=True):
    with open(filepath, "r") as f:
        lines = f.readlines()
    coords = []
    start = False
    for line in lines:
        if "NODE_COORD_SECTION" in line:
            start = True
            continue
        if start:
            if "EOF" in line:
                break
            parts = line.strip().split()
            if len(parts) >= 3:
                coords.append([float(parts[1]), float(parts[2])])
    coords = np.array(coords, dtype=np.float32)

    if normalize:
        # Scale all coordinates to [0, 1] by dividing by max value
        coords = coords / np.max(coords)

    return coords

def compute_distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(coords[i] - coords[j])
    return dist
