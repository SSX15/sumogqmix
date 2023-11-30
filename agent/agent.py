import pdb
import time

import torch
from utils.utils import zip_strict
from utils.replaybuffer import ReplayBuffer, EpisodeBuffer, EpisodeMemory
from model.QNetwork import QNetwork
from torch.optim import  Adam
from torch.nn import functional as F
from model.GATnet import GATNet
import numpy as np
class BaseAgent:
    def __init__(self, args):
        self.args = args
        self.rnn = args.rnn
        if self.rnn:
            from model.RNNQnet import RNNQnet
        #self.env = env
        self.agent_ids = args.agent_ids
        self.n_agent = len(self.agent_ids)
        self.state = args.init_state
        self.gl_s = args.init_gl_s
        self.action_space = args.action_space
        self.tl_ob_dim = self.args.tl_ob_dim
        self.n_ob = args.ob_dim + self.n_agent
        self.n_ac = self.action_space.n
        self.lr = args.lr
        self.epsilon_init = args.epsilon_init
        self.epsilon = self.epsilon_init
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        if self.rnn:
            self.buffer = EpisodeMemory(args)
        else:
            self.buffer = ReplayBuffer(args)
        self.load = args.load
        self.param_file = args.param_file
        self.device = self.args.device
        if not self.load:
            if self.args.gat:
                self.gat_net = GATNet(args)
                self.gat_net.to(self.device)
            if self.rnn:
                self.q_net = RNNQnet(self.n_ob, self.n_ac)
                self.q_net_target = RNNQnet(self.n_ob, self.n_ac)
            else:
                self.q_net = QNetwork(self.n_ob, self.n_ac)
                self.q_net_target = QNetwork(self.n_ob, self.n_ac)
            self.q_net_target.load_state_dict(self.q_net.state_dict())
        else:
            self.load_model()
            self.epsilon_init = 0
        self.q_net.to(self.device)
        self.q_net_target.to(self.device)
        self.q_net_target.eval()
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
        if self.rnn:
            self.rnn_train()
        else:
            self.no_rnn_train()
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
        if self.args.gat:
            batch['s'] = self.gat_net(batch['s'])
            batch['ns'] = self.gat_net(batch['ns'])
        for index in range(self.args.seq_len):
            #pdb.set_trace()
            input = batch['s'][:, index, :, :]
            input_n = batch['ns'][:, index, :, :]
            eye = torch.eye(self.n_agent).unsqueeze(0).expand(self.batch_size, -1, -1).to(self.device).to(torch.float32)
            input = torch.cat((input, eye), dim=2)
            input_n = torch.cat((input_n, eye), dim=2)

            q_v, train_hidden = self.q_net(input, train_hidden)
            next_q_v, train_target_hidden = self.q_net_target(input_n, train_target_hidden)

            q_v = q_v.view(self.batch_size, self.n_agent, -1)
            next_q_v = next_q_v.view(self.batch_size, self.n_agent, -1)
            q_vs.append(q_v)
            next_q_vs.append(next_q_v)

        q_vs = torch.stack(q_vs, dim=1)
        next_q_vs = torch.stack(next_q_vs, dim=1)
        q_vs = torch.gather(q_vs, dim=3, index=a.long()).squeeze(3)
        next_q_vs = next_q_vs.max(dim=3)[0]
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

            input = batch['s']
            input_n = batch['ns']
            if self.args.gat:
                input = self.gat_net(input.unsqueeze(1)).squeeze(1)
                input_n = self.gat_net(input_n.unsqueeze(1)).squeeze(1)
            eye = torch.eye(self.n_agent).unsqueeze(0).expand(self.batch_size, -1, -1).to(self.device).to(torch.float32)
            input = torch.cat((input, eye), dim=2)
            input_n = torch.cat((input_n, eye), dim=2)

            q_v = self.q_net(input)
            next_q_v = self.q_net_target(input_n).max(dim=2)[0]
            q_v = torch.gather(q_v, dim=2, index=a.long()).squeeze(2)

            target_q = r + self.gamma * next_q_v

            loss = F.smooth_l1_loss(q_v, target_q.detach())
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_gradient)
            self.optimizer.step()
        self.loss.append(np.mean(losses))

    def update_state(self, next_st, gl_ns):
        self.state = next_st
        self.gl_s = gl_ns
    def timestep_plus(self):
        self.timestep += 1

    def buffer_push(self, action, next_st, r, gl_ns):
        if self.rnn:
            transition = [self.state, action, next_st, r, self.gl_s, gl_ns]
            self.cur_episode.push(transition=transition)
        else:
            self.buffer.push(self.state, action, next_st, r, self.gl_s, gl_ns)

    def episode_push(self):
        self.buffer.push(self.cur_episode)
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
        if self.args.gat:
            input = self.gat_state[agent_id].cpu().numpy()
        else:
            input = self.state[agent_id]
        index = self.agent_ids.index(agent_id)
        index_hot = np.zeros(self.n_agent)
        index_hot[index] = 1
        input = np.hstack((input, index_hot))
        input = torch.as_tensor(input).to(self.device).unsqueeze(0).to(torch.float32)

        with torch.no_grad():
            if self.rnn:
                q_values, self.hidden_state[:, index, :] = self.q_net(input, self.hidden_state[:, index, :])
            else:
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
        if self.rnn:
            mergebatch = {'s': np.empty((self.batch_size, self.args.seq_len, self.n_agent, self.tl_ob_dim)),
                          'a': np.empty((self.batch_size, self.args.seq_len, self.n_agent, 1)),
                          'ns': np.empty((self.batch_size, self.args.seq_len, self.n_agent, self.tl_ob_dim)),
                          'r': np.empty((self.batch_size, self.args.seq_len, self.n_agent, 1)),
                          'gl_s': np.empty((self.batch_size, self.args.seq_len, self.args.gl_s_dim)),
                          'gl_ns': np.empty((self.batch_size, self.args.seq_len, self.args.gl_s_dim))}
            for i in range(self.batch_size):
                for seq_index in range(self.args.seq_len):
                    one_batch = batch[i]
                    for index, id in enumerate(self.agent_ids):
                        mergebatch['s'][i][seq_index][index] = one_batch['s'][seq_index][id]
                        mergebatch['a'][i][seq_index][index] = one_batch['a'][seq_index][id]
                        mergebatch['ns'][i][seq_index][index] = one_batch['ns'][seq_index][id]
                        mergebatch['r'][i][seq_index][index] = one_batch['r'][seq_index][id]
                    mergebatch['gl_s'][i][seq_index] = one_batch['gl_s'][seq_index]
                    mergebatch['gl_ns'][i][seq_index] = one_batch['gl_ns'][seq_index]
            mergebatch['s'] = torch.as_tensor(mergebatch['s']).to(self.device).to(torch.float32)
            mergebatch['a'] = torch.as_tensor(mergebatch['a']).to(self.device).to(torch.float32)
            mergebatch['ns'] = torch.as_tensor(mergebatch['ns']).to(self.device).to(torch.float32)
            mergebatch['r'] = torch.as_tensor(mergebatch['r']).to(self.device).to(torch.float32)
            mergebatch['gl_s'] = torch.as_tensor(mergebatch['gl_s']).to(self.device).to(torch.float32)
            mergebatch['gl_ns'] = torch.as_tensor(mergebatch['gl_ns']).to(self.device).to(torch.float32)
            return mergebatch
        else:
            mergebatch = {'s': np.empty((self.batch_size, self.n_agent, self.tl_ob_dim)),
                          'a': np.empty((self.batch_size, self.n_agent, 1)),
                          'ns': np.empty((self.batch_size, self.n_agent, self.tl_ob_dim)),
                          'r': np.empty((self.batch_size, self.n_agent, 1)),
                          'gl_s': np.empty((self.batch_size, self.args.gl_s_dim)),
                          'gl_ns': np.empty((self.batch_size, self.args.gl_s_dim))}
            for i in range(self.batch_size):
                one_batch = batch[i]
                for index, id in enumerate(self.agent_ids):
                    mergebatch['s'][i][index] = one_batch['s'][id]
                    mergebatch['a'][i][index] = one_batch['a'][id]
                    mergebatch['ns'][i][index] = one_batch['ns'][id]
                    mergebatch['r'][i][index] = one_batch['r'][id]
                mergebatch['gl_s'][i] = one_batch['gl_s']
                mergebatch['gl_ns'][i] = one_batch['gl_ns']
            mergebatch['s'] = torch.as_tensor(mergebatch['s']).to(self.device).to(torch.float32)
            mergebatch['a'] = torch.as_tensor(mergebatch['a']).to(self.device).to(torch.float32)
            mergebatch['ns'] = torch.as_tensor(mergebatch['ns']).to(self.device).to(torch.float32)
            mergebatch['r'] = torch.as_tensor(mergebatch['r']).to(self.device).to(torch.float32)
            mergebatch['gl_s'] = torch.as_tensor(mergebatch['gl_s']).to(self.device).to(torch.float32)
            mergebatch['gl_ns'] = torch.as_tensor(mergebatch['gl_ns']).to(self.device).to(torch.float32)
            return mergebatch

    def reset_st(self, state, gl_s):
        self.state = state
        self.gl_s = gl_s

    def init_rnn(self):
        self.hidden_state = self.init_hidden(1)
        self.target_hidden_state = self.init_hidden(1)
        self.cur_episode = EpisodeBuffer()

    def init_hidden(self, ep_num):
        return torch.zeros(ep_num, self.n_agent, 64).to(self.device).to(torch.float32)

    def get_gat_state(self, eval=False):
        gat_state = []
        for (k, v) in self.state.items():
            gat_state.append(v)
        gat_state = np.array(gat_state)
        gat_state = torch.from_numpy(gat_state).to(self.device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            gat_state = self.gat_net(gat_state, eval=eval)
        self.gat_state = self.state.copy()
        for i, (k, _) in enumerate(self.gat_state.items()):
            self.gat_state[k] = gat_state[0][0][i]

    def get_gat_state_eval(self, eval=True):
        gat_state = []
        for (k, v) in self.state.items():
            gat_state.append(v)
        gat_state = np.array(gat_state)
        gat_state = torch.from_numpy(gat_state).to(self.device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            gat_state = self.gat_net(gat_state, eval=eval)
        self.gat_state = self.state.copy()
        for i, (k, _) in enumerate(self.gat_state.items()):
            self.gat_state[k] = gat_state[0][0][i]

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
