import pdb

import torch.nn
from torch import nn
import collections
class DRQNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        #self.with_bias = True
        '''
        self.fc1 = nn.Linear(n_observations, 128, bias=self.with_bias)
        self.fc2 = nn.Linear(128, 64, bias=self.with_bias)
        self.fc3 = nn.Linear(64, 32, bias=self.with_bias)
        self.fc4 = nn.Linear(32, n_actions, bias=self.with_bias)
        '''
        self.obshape = n_observations

        self.hidden_size = 64
        self.net1 = nn.Sequential(collections.OrderedDict([
            ('fc1', nn.Linear(n_observations, 64)),
            ('relu1', nn.ReLU())
        ]))
        self.lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=True)
        self.net2 = nn.Sequential(collections.OrderedDict([
            ('fc2', nn.Linear(64, 32)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(32, n_actions))
            ]))

    def init_lstm(self, batch_size=None, device=None, train=False):
        if train:
            h = torch.zeros([1, batch_size, self.hidden_size], dtype=torch.float).to(device)
            c = torch.zeros([1, batch_size, self.hidden_size], dtype=torch.float).to(device)
        else:
            h = torch.zeros([1, 1, self.hidden_size], dtype=torch.float).to(device)
            c = torch.zeros([1, 1, self.hidden_size], dtype=torch.float).to(device)
        return h, c

    def forward(self, state, h, c):
        x = self.net1(state)
        x, (n_h, n_c) = self.lstm(x, (h, c))
        x = self.net2(x)
        return x, n_h, n_c



