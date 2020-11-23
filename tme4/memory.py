import numpy as np
from collections import deque
import torch

class ReplayBuffer(object):
    """Experience replay buffer that samples uniformly."""
    def __init__(self, size, device="cpu"):
        """Initializes the buffer."""
        self.buffer = deque(maxlen=size)
        self.device = device
        self.perm = None
        self.rewards = None
        self.dones = None

    def add(self, state, action, reward, next_state, done, succes):
        self.buffer.append((state, action, reward, next_state, done, succes))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        states, actions, rewards, next_states, dones, succes = [], [], [], [], [], []
        #idx = np.random.choice(len(self.buffer), num_samples)
        idx = np.arange(len(self.buffer))
        self.perm = idx
        
        for elem in self.buffer:
            #elem = self.buffer[i]
            state, action, reward, next_state, done, succ = elem
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
            succes.append(succ)
        
        states = torch.as_tensor(np.array(states), device=self.device)
        actions = torch.as_tensor(np.array(actions), device=self.device)
        self.rewards = torch.as_tensor(np.array(rewards, dtype=np.float32), device=self.device)
        next_states = torch.as_tensor(np.array(next_states), device=self.device)
        self.dones = torch.as_tensor(np.array(dones, dtype=np.float32), device=self.device)
        succes = torch.as_tensor(np.array(succes, dtype=np.float32), device=self.device)

        #import ipdb; ipdb.set_trace()

        return states[idx], actions[idx], self.rewards[idx], next_states[idx], self.dones[idx], succes[idx]


    def monte_carlo_sample(self,gamma):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        return rewards[self.perm]

class SumTree:
    def __init__(self, mem_size):
        self.tree = np.zeros(2 * mem_size - 1)
        self.data = np.zeros(mem_size, dtype=object)
        self.size = mem_size
        self.ptr = 0
        self.nentities=0


    def update(self, idx, p):
        tree_idx = idx + self.size - 1
        diff = p - self.tree[tree_idx]
        self.tree[tree_idx] += diff
        while tree_idx:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += diff

    def store(self, p, data):
        self.data[self.ptr] = data
        self.update(self.ptr, p)

        self.ptr += 1
        if self.ptr == self.size:
            self.ptr = 0
        self.nentities+=1
        if self.nentities > self.size:
            self.nentities = self.size


    def sample(self, value):
        ptr = 0
        while ptr < self.size - 1:
            left = 2 * ptr + 1
            if value < self.tree[left]:
                ptr = left
            else:
                value -= self.tree[left]
                ptr = left + 1

        return ptr - (self.size - 1), self.tree[ptr], self.data[ptr - (self.size - 1)]

    @property
    def total_p(self):
        return self.tree[0]

    @property
    def max_p(self):
        return np.max(self.tree[-self.size:])

    @property
    def min_p(self):
        return np.min(self.tree[-self.size:])


class Memory:

    def __init__(self, mem_size, prior=False,p_upper=1.,epsilon=.01,alpha=1,beta=1):
        self.p_upper=p_upper
        self.epsilon=epsilon
        self.alpha=alpha
        self.beta=beta
        self.prior = prior
        self.nentities=0
        #self.data_len = 2 * feature_size + 2
        if prior:
            self.tree = SumTree(mem_size)
        else:
            self.mem_size = mem_size
            self.mem = np.zeros(mem_size, dtype=object)
            self.mem_ptr = 0

    def store(self, transition):
        if self.prior:
            p = self.tree.max_p
            if not p:
                p = self.p_upper
            self.tree.store(p, transition)
        else:
            self.mem[self.mem_ptr] = transition
            self.mem_ptr += 1

            if self.mem_ptr == self.mem_size:
                self.mem_ptr = 0
            self.nentities += 1
            if self.nentities > self.mem_size:
                self.nentities = self.mem_size

    def sample(self, n):
        if self.prior:
            min_p = self.tree.min_p
            if min_p==0:
                min_p=self.epsilon**self.alpha
            seg = self.tree.total_p / n
            batch = np.zeros(n, dtype=object)
            w = np.zeros((n, 1), np.float32)
            idx = np.zeros(n, np.int32)
            a = 0
            for i in range(n):
                b = a + seg
                v = np.random.uniform(a, b)
                idx[i], p, batch[i] = self.tree.sample(v)

                w[i] = (p / min_p) ** (-self.beta)
                a += seg
            return idx, w, batch
        else:
            mask = np.random.choice(range(self.nentities), n)
            return self.mem[mask]

    def update(self, idx, tderr):
        if self.prior:
            tderr += self.epsilon
            tderr = np.minimum(tderr, self.p_upper)
            #print(idx,tderr)
            for i in range(len(idx)):
                self.tree.update(idx[i], tderr[i] ** self.alpha)

