import gym
import gridworld
import copy
import numpy as np
from collections import defaultdict
import utils as ut
import memory as mem
import torch
import pandas as pd
import torch.nn.functional as F



class DQNAgent(object):
    """Deep Qlearning"""

    def __init__(self, env, opt, test = False):

        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)

        self.lossFunc = torch.nn.SmoothL1Loss()
        self.targetStep = 2000
        self.batch_size = 32
        self.mem_size = 100000
        self.buffer = mem.ReplayBuffer(self.mem_size)
        self.test = test
        self.old_state = None
        self.old_act = None
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.qsa = ut.NN(self.featureExtractor.outSize, self.env.action_space.n, [32,32])
        self.optiQsa = torch.optim.Adam(params=self.qsa.parameters(),lr=1e-4)
        self.qsaTraget = copy.deepcopy(self.qsa)
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon0 = 1
        self.mu = 0.003

    def act(self, observation, reward, frame, episode, done):
        # Qtarget update
        if frame % self.targetStep == 0:
            self.setTarget(self.qsa)
        
        # Initialisation
        observation = torch.tensor(self.featureExtractor.getFeatures(observation), dtype=torch.float)
        qs = self.qsa(observation)

        if self.old_state == None:
            action = self.env.action_space.sample()
            self.old_state = observation
            self.old_act = action
            
            return action

        # epsilon greedy
        eps = self.epsilon0 / (1 + self.mu * episode)
        if np.random.rand() > eps or self.test == True:
            action = int(torch.argmax(qs))
        else:
            action = self.env.action_space.sample()
        

        # Remplissage du buffer
        self.buffer.add(self.old_state, self.old_act, reward, observation, done)
        
        # Apprentissage
        if len(self.buffer) > self.batch_size and self.test == False: # start training when there is enough example 
        #if len(self.buffer) == self.mem_size: # start training when buffer is full

            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            loss = self.train_step(states, actions, rewards, next_states, dones)
            loss.backward()
            self.optiQsa.step()
            self.optiQsa.zero_grad()

        # Update state and action
        self.old_state = observation
        self.old_act = action
            
        return action
    
    def train_step(self,states, actions, rewards, next_states, dones):
        """Perform a training iteration on a batch of data sampled from the experience
        replay buffer.
        """
        # Calculate targets.
        max_next_qs = self.qsaTraget(next_states).max(-1).values
        with torch.no_grad():
            target = rewards + (1.0 - dones) * self.gamma * max_next_qs
        qs = self.qsa(states)
        action_masks = F.one_hot(actions, self.env.action_space.n)
        masked_qs = (action_masks * qs).sum(dim=-1)
        loss = self.lossFunc(masked_qs, target.detach())
        #nn.utils.clip_grad_norm_(loss, max_norm=10)

        return loss


    def setTarget(self, target):
        self.qsaTraget = copy.deepcopy(target)
    
    def save(self,outputDir):
        pass
    
    def load(self,inputDir):
        pass



class PolicyGradAgent(object):
    """Policy Gradient agent"""

    def __init__(self, env, opt, test = False):

        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)

        self.lossFunc = torch.nn.MSELoss()
        self.targetStep = 1000
        self.batch_size = 100
        self.buffer = mem.ReplayBuffer(self.batch_size)
        self.test = test
        self.old_state = None
        self.old_act = None
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.vpi = ut.NN(self.featureExtractor.outSize, 1, [32,32]) # V value to estimate Actor
        self.policy = ut.NN(self.featureExtractor.outSize, env.action_space.n, [32]) # Policy 
        self.vpiTarget = self.vpi
        self.optiVpi = torch.optim.Adam(params=self.vpi.parameters(),lr=1e-3)
        self.optiPolicy = torch.optim.Adam(params=self.policy.parameters(),lr=1e-3)
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon0 = 1
        self.mu = 0.003

    def act(self, observation, reward, frame, episode, done):

        # Qtarget update
        if frame % self.targetStep == 0:
            self.setTarget(self.vpi) 
        
        # Initialisation
        observation = torch.tensor(self.featureExtractor.getFeatures(observation), dtype=torch.float)
        qs = self.policy(observation)

        if self.old_state == None:
            
            action = self.env.action_space.sample()
            self.old_state = observation
            self.old_act = action
            
            return action

        # epsilon greedy
        eps = self.epsilon0 / (1 + self.mu * episode)
        if np.random.rand() > eps or self.test == True:
            action = torch.argmax(qs)

        else:
            action = self.env.action_space.sample()

        # Remplissage du buffer
        self.buffer.add(self.old_state, self.old_act, reward, observation, done)

        # Apprentissage des Vpi
        if len(self.buffer) == self.batch_size:

            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            loss_vpi = self.train_step_vpi(states, actions, rewards, next_states, dones)
            loss_vpi.backward()
            self.optiVpi.step()
            self.optiVpi.zero_grad() # clear Vpi params gradient

            # Campute Actor
            actor = rewards + (1.0 - dones) * self.gamma * self.vpi(next_states) - self.vpi(states)

            # Compute reward expectation gradient

            # 1) compute expectation
            action_masks = F.one_hot(actions, self.env.action_space.n)
            masked_proba = (action_masks * self.policy(states)).sum(dim=-1)
            logProba = torch.log(masked_proba)
            expect = torch.sum(logProba*actor)

            # 2) back propagation
            expect.backward()

            # 3) Update parameters and reset buffer
            self.optiPolicy.step()
            self.optiPolicy.zero_grad() # clear Policy params gradient
            self.resetBuffer() # reset buffer to sample with next step policy

        # Update state and action
        self.old_state = observation
        self.old_act = action
            
        return action

    def train_step_vpi(self,states, actions, rewards, next_states, dones):
        """Perform a training iteration on a batch of data sampled from the experience
        replay buffer.
        """
        # Calculate targets.
        vpi = self.vpi(states)
        with torch.no_grad():
            next_vpi = self.vpiTarget(next_states)
            target = rewards + (1.0 - dones) * self.gamma * next_vpi
        loss_vpi = self.lossFunc(vpi, target.detach())

        return loss_vpi


    def setTarget(self, target):
        self.vpiTarget = copy.deepcopy(target)

    def resetBuffer(self):
        self.buffer = mem.ReplayBuffer(self.batch_size)
    
    def save(self,outputDir):
        pass
    
    def load(self,inputDir):
        pass