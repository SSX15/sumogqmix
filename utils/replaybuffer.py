import random
import numpy as np
import collections

class ReplayBuffer:
    def __init__(self, args):
        self.buffer_size = args.buffer_size
        #self.state_shape = state_space.shape
        #self.memory = np.zeros((self.buffer_size, *self.state_shape), state_space.dtype)
        self.memory = []
        self.pos = 0
    def push(self, state, aciton, nextstate, reward):
        transition = {'s': state, 'a': aciton, 'ns': nextstate, 'r': reward}
        if self.pos > self.buffer_size:
            curpos = self.pos % self.buffer_size
            self.memory[curpos].update(transition)
        else:
            self.memory.append(transition)
        self.pos += 1

    def sample(self, batchsize):
        return random.sample(self.memory, batchsize)

    def len(self):
        return len(self.memory)

    def full(self):
        return self.pos >= self.buffer_size

class ReplayBuffer2:
    def __init__(self, args):
        self.buffer_size = args.buffer_size
        self.n_agent = len(args.agent_ids)
        self.ob_space = args.ob_space
        #self.memory = np.zeros((self.buffer_size, *self.state_shape), state_space.dtype)
        self.memory = {
            "s": np.empty([self.buffer_size, self.n_agent, self.ob_space]),
            "a": np.empty([self.buffer_size, self.n_agent, 1]),
            "n_s": np.empty([self.buffer_size, self.n_agent, self.ob_space]),
            "r": np.empty([self.buffer_size, self.n_agent, 1])
        }
        self.pos = 0
    def push(self, state, aciton, nextstate, reward):
        transition = {'s': state, 'a': aciton, 'ns': nextstate, 'r': reward}
        if self.pos > self.buffer_size:
            curpos = self.pos % self.buffer_size
            self.memory[curpos].update(transition)
        else:
            self.memory.append(transition)
        self.pos += 1

    def sample(self, batchsize):
        return random.sample(self.memory, batchsize)

    def len(self):
        return len(self.memory)

    def full(self):
        return self.pos >= self.buffer_size


class EpisodeMemory:
    def __init__(self, max_epi_len=3600, max_mem_size=10, batch_size=1, seq_len=1):
        self.max_epi_len = max_epi_len
        self.max_mem_size = max_mem_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.memory = collections.deque(maxlen=self.max_mem_size)

    def push(self, episode):
        self.memory.append(episode)
    def sample(self):
        ep_seq_buf = []
        sampled_eps = random.choices(self.memory, k=self.batch_size)
        for ep in sampled_eps:
            idx = np.random.randint(0, ep.len()-self.seq_len+1)
            ep_seq = ep.sample(idx, self.seq_len)
            ep_seq_buf.append(ep_seq)
        return ep_seq_buf
    def len(self):
        return len(self.memory)

class EpisodeBuffer:
    def __init__(self):
        self.s = []
        self.a = []
        self.r = []
        self.n_s = []
        #self.done = []

    def push(self, transition):
        self.s.append(transition[0])
        self.a.append(transition[1])
        self.r.append(transition[2])
        self.n_s.append(transition[3])
        #self.done.append(transition[4])

    def sample(self, idx, seq_len):
        s = np.array(self.s)
        a = np.array(self.a)
        r = np.array(self.r)
        n_s = np.array(self.n_s)
        #done = np.array(self.done)

        s = s[idx:idx+seq_len]
        a = a[idx:idx+seq_len]
        r = r[idx:idx+seq_len]
        n_s = n_s[idx:idx+seq_len]
        #done = done[idx:idx+seq_len]
        return dict(s=s, a=a, r=r, n_s=n_s)

    def len(self):
        return len(self.s)