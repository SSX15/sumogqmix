import torch
import torch.nn as nn

class VDNnet(nn.Module):
    def __init__(self):
        super(VDNnet, self).__init__()
    def forward(self, q_values, agent_dim=1):
        return torch.sum(q_values, dim=agent_dim, keepdim=True)
