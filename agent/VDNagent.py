import pdb
import time

import torch
from utils.utils import zip_strict
from utils.replaybuffer import ReplayBuffer
from model.QNetwork import  QNetwork
from torch.optim import  Adam
from torch.nn import functional as F
from model.VDNnet import VDNnet

import numpy as np
class Agent:
    def __init__(self, args):
        #self.env = env
        self.agent_ids = args.agent_ids
        self.n_agent = len(self.agent_ids)
        self.state = args.init_state
        self.ob_space = args.ob_space
        self.action_space = args.action_space
        self.n_ob = self.ob_space.shape[0]
        self.n_ac = self.action_space.n
        self.lr = args.lr
        self.epsilon_init = args.epsilon_init
        self.epsilon = self.epsilon_init
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.buffer = ReplayBuffer(args)
        self.load = args.load
        self.param_file = args.param_file
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if not self.load:
            self.q_net = QNetwork(self.n_ob, self.n_ac)
            self.q_net_target = QNetwork(self.n_ob, self.n_ac)
            self.vdn_net = VDNnet()
            self.vdn_net_target = VDNnet()
            self.q_net_target.load_state_dict(self.q_net.state_dict())
            self.vdn_net_target.load_state_dict(self.vdn_net.state_dict())
        else:
            self.load_model()
            self.epsilon_init = 0
        self.q_net.to(self.device)
        self.q_net_target.to(self.device)
        self.q_net_target.eval()
        self.vdn_net.to(self.device)
        self.vdn_net_target.to(self.device)
        self.optimizer = Adam(self.q_net.parameters(), lr=args.lr)
        self.tau = 1
        self.timestep = 0
        self.gradient_step = args.gradient_step
        self.train_freq = args.train_freq
        self.target_update_freq = args.target_update_freq
        self.max_gradient = 10
        self.loss = []
        self.start_size = args.start_size


    def train(self):
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

            #input, input_n = [], []
            input = batch['s']
            input_n = batch['ns']
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

    def update_state(self, next_st):
        self.state = next_st
    def timestep_plus(self):
        self.timestep += 1

    def buffer_push(self, action, next_st, r):
        self.buffer.push(self.state, action, next_st, r)
    def collect_more_step(self, train_freq, num_coll_steps):
        return num_coll_steps < train_freq

    def choose_action(self, agent_id):
        action = self.predict(agent_id)
        #action = action.numpy().reshape(-1)
        return action

    def update_epsilon(self, episode, max_episode):
        episode_threshold = 0.1 * max_episode
        if episode < episode_threshold:
            self.epsilon = self.epsilon_init * (1 - episode / episode_threshold)
        else:
            self.epsilon = 0

    def predict(self, agent_id):
        input = self.state[agent_id]
        index = self.agent_ids.index(agent_id)
        index_hot = np.zeros(self.n_agent)
        index_hot[index] = 1
        input = np.hstack((input, index_hot))
        input = torch.as_tensor(input).to(self.device).unsqueeze(0).to(torch.float32)

        with torch.no_grad():
            q_values = self.q_net(input)

        if np.random.rand() < self.epsilon:
            action = self.action_space.sample()
        else:
            #print(q_values)
            action = q_values.argmax(dim=1).reshape(-1)
            #print(action)
        return action

    def update_target(self):
        if self.timestep % self.target_update_freq == 0:
            with torch.no_grad():
                for param, target_param in zip_strict(self.q_net.parameters(), self.q_net_target.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
                    #self.update_epsilon()

    def merge_batch(self, batch):
        mergebatch = {'s': np.empty((self.batch_size, self.n_agent, (self.n_ob - self.n_agent))),
                      'a': np.empty((self.batch_size, self.n_agent, 1)),
                      'ns': np.empty((self.batch_size, self.n_agent, self.n_ob - self.n_agent)),
                      'r': np.empty((self.batch_size, self.n_agent, 1))}
        for i in range(self.batch_size):
            one_batch = batch[i]
            for index, id in enumerate(self.agent_ids):
                mergebatch['s'][i][index] = one_batch['s'][id]
                mergebatch['a'][i][index] = one_batch['a'][id]
                mergebatch['ns'][i][index] = one_batch['ns'][id]
                mergebatch['r'][i][index] = one_batch['r'][id]
        mergebatch['s'] = torch.as_tensor(mergebatch['s'])
        mergebatch['a'] = torch.as_tensor(mergebatch['a'])
        mergebatch['ns'] = torch.as_tensor(mergebatch['ns'])
        mergebatch['r'] = torch.as_tensor(mergebatch['r'])
        return mergebatch

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
