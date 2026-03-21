# src/agent.py
import torch
from torch.distributions import Categorical

class GRPOAgent:
    def __init__(self, model, lr=1e-4, device="cpu"):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = device

    def act(self, x, edge_index, visited_mask=None, current_idx=None, deterministic=False):
        """
        x: node features tensor (n_nodes, feature_dim)
        edge_index: graph connectivity (2, E)
        visited_mask: boolean mask tensor (n_nodes,) indicating visited nodes (optional)
        current_idx: integer index of current node (optional, not used here)
        deterministic: if True, select argmax; else sample from policy
        returns: action (int), log_prob (tensor scalar)
        """
        logits = self.model(x.to(self.device), edge_index.to(self.device))
        # Mask visited nodes if provided
        if visited_mask is not None:
            logits = logits.clone()
            logits[visited_mask.bool()] = -1e9
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs).unsqueeze(0)
            # Create a dummy log_prob consistent with sampling shape
            log_prob = torch.log(probs[action])
            return action.item(), log_prob
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
