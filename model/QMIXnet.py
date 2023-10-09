import torch
import torch.nn as nn
import torch.nn.functional as F

class QMIXnet(nn.Module):
    def __init__(self, args):
        super(QMIXnet, self).__init__()
        self.agent_ids = args.agent_ids
        self.n_agent = len(self.agent_ids)
        self.ob_space = args.ob_space
        self.qmix_hidden_dim = 64
        self.state_shape = self.ob_space.shape[0] * self.n_agent
        #超网络的两层全连接参数
        self.hyper_w1 = nn.Linear(self.state_shape, self.n_agent * self.qmix_hidden_dim)
        self.hyper_w2 = nn.Linear(self.state_shape, self.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(self.state_shape, self.qmix_hidden_dim)
        self.hyper_b2 =nn.Sequential(nn.Linear(self.state_shape, self.qmix_hidden_dim * 1),
                                     nn.ReLU(),
                                     nn.Linear(self.qmix_hidden_dim, 1))
    def forward(self, q_values, state): #q_v:(batch, n_agent)  state:(batch, state_shape)
        q_values = q_values.view(-1, 1, self.n_agent) #(batch, 1, n_agent)


        h_w1 = torch.abs(self.hyper_w1(state)) #(batch, n_agent * 32)
        h_w2 = torch.abs(self.hyper_w2(state)) #(batch, 32 * 1)

        h_w1 = h_w1.view(-1, self.n_agent, self.qmix_hidden_dim) #(batch, n_agent, 32)
        h_w2 = h_w2.view(-1, self.qmix_hidden_dim, 1) #(batch, 32, 1)

        h_b1 = self.hyper_b1(state) #(batch, 32)
        h_b2 = self.hyper_b2(state) #(batch, 1)

        h_b1 = h_b1.view(-1, 1, self.qmix_hidden_dim)
        h_b2 = h_b2.view(-1, 1, 1)

        hidden = F.elu(torch.bmm(q_values, h_w1) + h_b1) #(batch, 1, 32)
        q_total = F.elu(torch.bmm(hidden, h_w2) + h_b2) #(batch, 1, 1)
        q_total = q_total.squeeze(1)
        return q_total
