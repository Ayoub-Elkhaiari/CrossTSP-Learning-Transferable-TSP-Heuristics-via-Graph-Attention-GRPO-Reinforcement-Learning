# src/environment.py
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class TSPEnv(gym.Env):
    """
    TSP environment:
    - coords: numpy array (n,2) with coordinates (not necessarily normalized)
    - Observations: dict with 'coords' (normalized), 'visited' (0/1 mask), 'current' (int)
    - step(action) -> obs, reward, done, info
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, coords, render_mode=False):
        super().__init__()
        self.coords = np.asarray(coords, dtype=np.float32)
        self.n = len(coords)
        # Precompute full distance matrix
        self.dist = np.linalg.norm(self.coords[:, None, :] - self.coords[None, :, :], axis=-1)
        self.action_space = spaces.Discrete(self.n)
        # observation: coords (n,2), visited (n,), current (int)
        self.observation_space = spaces.Dict({
            "coords": spaces.Box(low=0.0, high=1.0, shape=(self.n, 2), dtype=np.float32),
            "visited": spaces.MultiBinary(self.n),
            "current": spaces.Discrete(self.n),
        })

        # Rendering
        self.render_mode = render_mode
        self.fig, self.ax = None, None
        self.line = None

    def reset(self, seed=None, options=None):
        # Reset visited and choose a random start
        self.visited = np.zeros(self.n, dtype=np.int32)
        self.current = np.random.randint(self.n)
        self.visited[self.current] = 1
        self.tour = [int(self.current)]
        # For agent input we normalize coords to [0,1] by global max
        max_val = np.max(self.coords)
        if max_val == 0:
            self.coords_norm = self.coords.copy()
        else:
            self.coords_norm = self.coords / max_val
        if self.render_mode:
            self._init_render()
        return self._get_obs()

    def _get_obs(self):
        return {
            "coords": self.coords_norm.copy(),
            "visited": self.visited.copy(),
            "current": int(self.current)
        }

    def step(self, action):
        action = int(action)
        info = {}
        if self.visited[action]:
            reward = -10.0  # penalty for invalid move
        else:
            reward = -self.dist[self.current, action]
            self.current = action
            self.visited[action] = 1
            self.tour.append(int(action))

        done = bool(self.visited.sum() == self.n)
        if self.render_mode:
            self.render()
        return self._get_obs(), float(reward), done, info

    # ---------- Rendering logic ----------
    def _init_render(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title("TSP Agent Path")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)
        xs, ys = self.coords[:, 0], self.coords[:, 1]
        self.ax.scatter(xs, ys, c="blue", s=40)
        for i, (x, y) in enumerate(self.coords):
            self.ax.text(x + 0.01, y + 0.01, str(i + 1), fontsize=8)
        self.line, = self.ax.plot([], [], "r-", linewidth=2)
        plt.pause(0.001)

    def render(self, mode="human"):
        if self.fig is None or self.ax is None:
            self._init_render()
        if len(self.tour) > 1:
            path = self.coords[self.tour]
            self.line.set_data(path[:, 0], path[:, 1])
            self.ax.relim()
            self.ax.autoscale_view()
        plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.ioff()
            plt.close(self.fig)
            self.fig, self.ax = None, None
