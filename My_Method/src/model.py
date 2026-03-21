import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GATPolicy(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, heads=4):
        super().__init__()
        # First layer: 4 attention heads, outputs hidden_dim * 4 features
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4)
        # Second layer: aggregate to hidden_dim with single head
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1)
        self.policy_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        logits = self.policy_head(x)
        return logits.squeeze(-1)
