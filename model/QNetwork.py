import pdb

from torch import nn
import collections
class QNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        #self.with_bias = True
        '''
        self.fc1 = nn.Linear(n_observations, 128, bias=self.with_bias)
        self.fc2 = nn.Linear(128, 64, bias=self.with_bias)
        self.fc3 = nn.Linear(64, 32, bias=self.with_bias)
        self.fc4 = nn.Linear(32, n_actions, bias=self.with_bias)
        '''
        self.net = nn.Sequential(collections.OrderedDict([
            ('fc2', nn.Linear(n_observations, 64)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(64, 32)),
            ('relu3', nn.ReLU()),
            ('fc4', nn.Linear(32, n_actions))
            ]))
    def forward(self, state):
        return self.net(state)


