import pdb
import time
import torch
from utils.utils import zip_strict
from torch.nn import functional as F
from model.VDNnet import VDNnet
from agent.agent import BaseAgent

import numpy as np
class Agent(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        #self.env = env
        if not self.load:
            self.vdn_net = VDNnet()
            self.vdn_net_target = VDNnet()
            self.vdn_net_target.load_state_dict(self.vdn_net.state_dict())
        else:
            self.load_model()
            self.epsilon_init = 0
        self.vdn_net.to(self.device)
        self.vdn_net_target.to(self.device)

    def rnn_train(self):
        if (self.buffer.len()) < 1:
            return
        losses = []
        train_hidden = self.init_hidden(self.batch_size)
        train_target_hidden = self.init_hidden(self.batch_size)
        batch = self.buffer.sample()
        batch = self.merge_batch(batch)
        q_vs, next_q_vs = [], []
        a = batch['a'].to(self.device)
        r = batch['r'].to(self.device).squeeze(3)
        r = torch.sum(r, dim=2).reshape(self.batch_size, -1, 1)
        #r = torch.mean(r, dim=2).reshape(self.batch_size, -1, 1)
        if self.args.gat:
            batch['s'] = self.gat_net(batch['s'])
            batch['ns'] = self.gat_net(batch['ns'])
        for index in range(self.args.seq_len):
            # pdb.set_trace()
            input = batch['s'][:, index, :, :]
            input_n = batch['ns'][:, index, :, :]
            eye = torch.eye(self.n_agent).unsqueeze(0).expand(self.batch_size, -1, -1)
            input = torch.cat((input, eye), dim=2).to(self.device).to(torch.float32)
            input_n = torch.cat((input_n, eye), dim=2).to(self.device).to(torch.float32)

            q_v, train_hidden = self.q_net(input, train_hidden)
            next_q_v, train_target_hidden = self.q_net_target(input_n, train_target_hidden)

            q_v = q_v.view(self.batch_size, self.n_agent, -1)
            next_q_v = next_q_v.view(self.batch_size, self.n_agent, -1)
            q_vs.append(q_v)
            next_q_vs.append(next_q_v)

        q_vs = torch.stack(q_vs, dim=1)
        next_q_vs = torch.stack(next_q_vs, dim=1)
        q_vs = torch.gather(q_vs, dim=3, index=a.long()).squeeze(3)
        q_vs = self.vdn_net(q_vs, 2)
        next_q_vs = next_q_vs.max(dim=3)[0]
        next_q_vs = self.vdn_net_target(next_q_vs, 2)
        target_q = r + self.gamma * next_q_vs

        loss = F.smooth_l1_loss(q_vs, target_q.detach())
        losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_gradient)
        self.optimizer.step()
        self.loss.append(np.mean(losses))
    def no_rnn_train(self):
        if (self.buffer.len()) < self.start_size:
            return
        losses = []
        for _ in range(self.gradient_step):
            #pdb.set_trace()
            batch = self.buffer.sample(self.batch_size)
            batch = self.merge_batch(batch)
            a = batch['a'].to(self.device)
            r = batch['r'].to(self.device).squeeze(2)
            r = torch.sum(r, dim=1).reshape(-1, 1)
            #r = torch.mean(r, dim=1).reshape(-1, 1)

            input = batch['s']
            input_n = batch['ns']
            if self.args.gat:
                input = self.gat_net(input.unsqueeze(1)).squeeze(1)
                input_n = self.gat_net(input_n.unsqueeze(1)).squeeze(1)
            eye = torch.eye(self.n_agent).unsqueeze(0).expand(self.batch_size, -1, -1)
            input = torch.cat((input, eye), dim=2).to(self.device).to(torch.float32)
            input_n = torch.cat((input_n, eye), dim=2).to(self.device).to(torch.float32)

            q_v = self.q_net(input)
            next_q_v = self.q_net_target(input_n).max(dim=2)[0]
            q_v = torch.gather(q_v, dim=2, index=a.long()).squeeze(2)

            q_v = self.vdn_net(q_v)
            next_q_v = self.vdn_net_target(next_q_v)

            target_q = r + self.gamma * next_q_v

            loss = F.smooth_l1_loss(q_v, target_q.detach())
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_gradient)
            self.optimizer.step()
        self.loss.append(np.mean(losses))

    def update_target(self):
        if self.timestep % self.target_update_freq == 0:
            with torch.no_grad():
                for param, target_param in zip_strict(self.q_net.parameters(), self.q_net_target.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
                for param, target_param in zip_strict(self.vdn_net.parameters(), self.vdn_net_target.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

    def reset_st(self, state):
        self.state = state


    def save_parameters(self):
        qnet_path = self.param_file + '{}_qnet.pt'.format(self.agent_id)
        target_path = self.param_file + '{}_target.pt'.format(self.agent_id)
        torch.save(self.q_net.state_dict(), qnet_path)
        torch.save(self.q_net_target.state_dict(), target_path)

    def load_model(self):
        qnet_path = self.param_file + '{}_qnet.pt'.format(self.agent_id)
        target_path = self.param_file + '{}_target.pt'.format(self.agent_id)
        self.q_net = QNetwork(self.n_ob, self.n_ac)
        self.q_net_target = QNetwork(self.n_ob, self.n_ac)
        self.q_net.load_state_dict(torch.load(qnet_path, map_location=torch.device('cpu')))
        self.q_net_target.load_state_dict(torch.load(target_path, map_location=torch.device('cpu')))
