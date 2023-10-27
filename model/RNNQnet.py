import pdb
import torch
from torch import nn
import collections
class RNNQnet(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()

        self.hidden_size = 64
        self.net1 = nn.Sequential(collections.OrderedDict([
            ('fc1', nn.Linear(n_observations, 64)),
            ('relu1', nn.ReLU())
        ]))
        self.rnn = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.net2 = nn.Linear(self.hidden_size, n_actions)

    def forward(self, state, hidden_state):
        x = self.net1(state)
        x = x.view(-1, self.hidden_size)
        hidden_state = hidden_state.view(-1, self.hidden_size)
        h = self.rnn(x, hidden_state)
        q = self.net2(h)
        return q, h



