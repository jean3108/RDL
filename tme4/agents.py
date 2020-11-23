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
from torch.distributions import Categorical



class DQNAgent(object):
    """Deep Qlearning"""

    def __init__(self, env, opt, episodes, batch_size, target_step, dim_layers, num_layers, test = False):
        self.nb_episodes = episodes
        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)

        self.lossFunc = torch.nn.SmoothL1Loss()
        self.targetStep = target_step
        self.batch_size = batch_size
        self.dim_layers = dim_layers
        self.num_layers = num_layers
        self.mem_size = 10000
        self.buffer = mem.ReplayBuffer(self.mem_size)
        self.test = test
        self.old_state = None
        self.old_act = None
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.qsa = ut.NN(self.featureExtractor.outSize, self.env.action_space.n, [dim_layers for i in range(num_layers)])
        self.optiQsa = torch.optim.Adam(params=self.qsa.parameters(),lr=1e-4)
        self.qsaTraget = copy.deepcopy(self.qsa)
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon0 = 1
        self.mu = 10/self.nb_episodes
        

    def act(self, observation, reward, frame, episode, done, succ):
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
        self.buffer.add(self.old_state, self.old_act, reward, observation, done, succ)
        
        # Apprentissage
        #if len(self.buffer) > self.batch_size and self.test == False: # start training when there is enough example 
        if len(self.buffer) == self.mem_size and self.test == False: # start training when buffer is full

            states, actions, rewards, next_states, dones, succes = self.buffer.sample(self.batch_size)
            loss = self.train_step(states, actions, rewards, next_states, dones, succes)
            loss.backward()
            self.optiQsa.step()
            self.optiQsa.zero_grad()

        # Update state and action
        self.old_state = observation
        self.old_act = action
            
        return action
    
    def train_step(self,states, actions, rewards, next_states, dones, succes):
        """Perform a training iteration on a batch of data sampled from the experience
        replay buffer.
        """
        # Calculate targets.
        max_next_qs = self.qsaTraget(next_states).max(-1).values
        with torch.no_grad():
            target = rewards + (1.0 - dones + succes) * self.gamma * max_next_qs
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
    """Policy Gradient agent -> Actor critic or PPO"""

    def __init__(self, env, opt, episodes, batch_size, target_step, dim_layers, num_layers, lr1, lr2, mu=10, test = False, loss_Func = 0, mode = 'actor-critic'):
        self.mode = mode
        self.nb_episodes = episodes
        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)

        self.lossFuncs = [torch.nn.MSELoss(), torch.nn.SmoothL1Loss()]
        self.lossFunc = self.lossFuncs[loss_Func]
        self.targetStep = target_step
        self.batch_size = batch_size
        self.dim_layers = dim_layers
        self.num_layers = num_layers
        self.buffer = mem.ReplayBuffer(self.batch_size)
        self.test = test
        self.old_state = None
        self.old_act = None
        self.old_prob = None
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.vpi = ut.NN(self.featureExtractor.outSize, 1, [dim_layers for i in range(num_layers)]) # state value
        self.policy = ut.Policy_NN(self.featureExtractor.outSize, env.action_space.n, [dim_layers for i in range(num_layers)]) # Policy 
        #self.net = ut.Policy_NN_2(self.featureExtractor.outSize, env.action_space.n)
        #self.optim = torch.optim.Adam(params=self.net.parameters(),lr=1e-2)
        self.vpiTarget = copy.deepcopy(self.vpi)
        #self.netTarget = copy.deepcopy(self.net)
        self.optiVpi = torch.optim.Adam(params=self.vpi.parameters(),lr=lr1)
        self.optiPolicy = torch.optim.Adam(params=self.policy.parameters(),lr=lr2)
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon0 = 1
        self.opti_count = 0
        self.mu = mu/self.nb_episodes
        self.eps_clip = 0.2
        self.k_step = 5
        self.step_critic = 0

    def act(self, observation, reward, frame, episode, done, succ):

        # Initialisation
        observation = torch.tensor(self.featureExtractor.getFeatures(observation), dtype=torch.float)
        distrib = Categorical(self.policy(observation))
        
        if self.old_state == None:
            #action = self.env.action_space.sample()
            action = int(distrib.sample())
            self.old_state = observation
            self.old_act = action
            
            return action
        action = int(distrib.sample())
        
        # Fill buffer
        self.buffer.add(self.old_state, self.old_act, reward, observation, done, succ)

        # Start learning
        if len(self.buffer) == self.batch_size and self.test == False:
            # collect samples
            states, actions, rewards, next_states, dones, succes = self.buffer.sample(self.batch_size)
            action_masks = F.one_hot(actions, self.env.action_space.n)
            probas = self.policy(states)

            if self.mode == "actor-critic":
                advantage_mode = 0
                advantages, critic_loss = self.avantage(states, actions, rewards, next_states, dones, mode=advantage_mode)
                #import ipdb; ipdb.set_trace()
                masked_log_proba = (action_masks * torch.log(probas)).sum(dim=-1)
                actor_loss = torch.mean(masked_log_proba*advantages.detach())
                #import ipdb; ipdb.set_trace()
                self.optiPolicy.zero_grad()
                actor_loss.backward()
                self.optiPolicy.step()
                
                if critic_loss != None:
                    self.optiVpi.zero_grad()
                    critic_loss.backward()
                    self.optiVpi.step()

            elif self.mode == "PPO":
                advantage_mode = 0
                
                for _ in range(self.k_step):

                    advantages, critic_loss = self.avantage(states, actions, rewards, next_states, dones, mode=advantage_mode)

                    ratios = self.policy(states) / probas.detach()
                    masked_ratios = (action_masks * ratios).sum(dim=-1)
                    t1 = masked_ratios * advantages.detach()
                    t2 = torch.clamp(masked_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages.detach()
                    actor_loss = torch.mean(-torch.min(t1,t2))
                    #import ipdb; ipdb.set_trace()

                    # policy step
                    self.optiPolicy.zero_grad() 
                    actor_loss.backward()
                    self.optiPolicy.step()

                    self.optiVpi.zero_grad()
                    critic_loss.backward()
                    self.optiVpi.step()

                

            if self.opti_count % self.targetStep == 0:
                self.setTarget(self.vpi)
            self.opti_count += 1
            self.resetBuffer() # reset buffer to sample with next step policy

        # Update state and action
        self.old_state = observation
        self.old_act = action
            
        return action


    def avantage(self, states, actions, rewards, next_states, dones, mode):
        if mode == 0:
            vpi = self.vpi(states).squeeze()
            with torch.no_grad():
                next_vpi = self.vpiTarget(next_states).squeeze()
                target = rewards + (1.0 - dones) * self.gamma * next_vpi
            return target - vpi, self.lossFunc(vpi, target.detach())

        elif mode == 1:
            R = self.buffer.monte_carlo_sample(self.gamma)
            #R = (R - R.mean()) /(R.std() + 1e-5)
            vpi = self.vpi(states).squeeze()
            return R - vpi, self.lossFunc(vpi, R.detach())

        elif mode == 2:
            R = self.buffer.monte_carlo_sample(self.gamma)
            return R - R.mean(), None

    def setTarget(self, target):
        self.vpiTarget = copy.deepcopy(target)

    def resetBuffer(self):
        self.buffer = mem.ReplayBuffer(self.batch_size)
    
    def save(self,outputDir):
        pass
    
    def load(self,inputDir):
        pass