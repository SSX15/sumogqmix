import torch
import torch.nn as nn

class VDNnet(nn.Module):
    def __init__(self):
        super(VDNnet, self).__init__()
    def forward(self, q_values):
        return torch.sum(q_values, dim=1, keepdim=True)
