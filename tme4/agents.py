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
        self.vpi = ut.NN(self.featureExtractor.outSize, 1, [dim_layers for i in range(num_layers)])
        self.policy = ut.Policy_NN(self.featureExtractor.outSize, env.action_space.n, [dim_layers for i in range(num_layers)])
        self.vpiTarget = copy.deepcopy(self.vpi)
        self.optiVpi = torch.optim.Adam(params=self.vpi.parameters(),lr=lr1)
        self.optiPolicy = torch.optim.Adam(params=self.policy.parameters(),lr=lr2)
        self.gamma = 0.99
        self.opti_count = 0
        self.eps_clip = 0.2
        self.k_step = 1
        self.step_critic = 0
        self.act_loss = None
        self.critic_loss = None
        self.loss_computed = False

    def act(self, observation, reward, frame, episode, done, succ):

        # Initialisation
        observation = torch.tensor(self.featureExtractor.getFeatures(observation), dtype=torch.float)
        distrib = Categorical(self.policy(observation))
        
        if self.old_state == None:
            action = int(distrib.sample())
            self.old_state = observation
            self.old_act = action
            
            return action

        # sample action from current policy
        action = int(distrib.sample())
        
        # Fill buffer
        self.buffer.add(self.old_state, self.old_act, reward, observation, done, succ)

        # Update state and action
        self.old_state = observation
        self.old_act = action

        # Start Training
        if len(self.buffer) == self.batch_size and self.test == False:

            # collect samples
            states, actions, rewards, next_states, dones, succes = self.buffer.sample(self.batch_size)
            action_masks = F.one_hot(actions, self.env.action_space.n)
            probas = self.policy(states)

            if self.mode == "actor-critic":
                advantage_mode = 0

                # compute advantages (and critic loss if advantage_mode == 0 ou 1) 
                advantages, critic_loss = self.avantage(states, actions, rewards, next_states, dones, succes, mode=advantage_mode)
                
                # Compute probas and actor_loss
                masked_log_proba = (action_masks * torch.log(probas)).sum(dim=-1)
                actor_loss = torch.mean(masked_log_proba*advantages.detach())
                
                self.optiPolicy.zero_grad()
                actor_loss.backward()
                self.optiPolicy.step()
                
                if critic_loss != None:
                    #import ipdb; ipdb.set_trace()
                    self.optiVpi.zero_grad()
                    critic_loss.backward()
                    self.optiVpi.step()

                self.setLosses(actor_loss, critic_loss)

            elif self.mode == "PPO":
                advantage_mode = 0
                advantages, critic_loss = self.avantage(states, actions, rewards, next_states, dones, succes, mode=advantage_mode)

                act_loss_sum = 0
                for _ in range(self.k_step):

                    ratios = self.policy(states) / probas.detach()
                    masked_ratios = (action_masks * ratios).sum(dim=-1)
                    t1 = masked_ratios * advantages.detach()
                    t2 = torch.clamp(masked_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages.detach()
                    actor_loss = torch.mean(torch.min(t1,t2))
                    act_loss_sum += actor_loss

                    # policy step
                    self.optiPolicy.zero_grad() 
                    actor_loss.backward()
                    self.optiPolicy.step()

                # Critic step
                self.optiVpi.zero_grad()
                critic_loss.backward()
                self.optiVpi.step()

                self.setLosses(act_loss_sum/self.k_step, critic_loss)

                

            if self.opti_count % self.targetStep == 0:
                self.setTarget(self.vpi)
            self.opti_count += 1
            self.resetBuffer() # reset buffer to sample with next step policy

        return action


    def avantage(self, states, actions, rewards, next_states, dones, succes, mode):

        # Actor-critic TD0
        if mode == 0:
            vpi = self.vpi(states).squeeze()
            with torch.no_grad():
                next_vpi = self.vpiTarget(next_states).squeeze()
                target = rewards + (1.0 - dones + succes) * self.gamma * next_vpi
            return target - vpi, self.lossFunc(vpi, target.detach())

        # Actor-critic with baseline
        elif mode == 1:
            R = self.buffer.monte_carlo_sample(self.gamma)
            #R = (R - R.mean()) /(R.std() + 1e-5)
            vpi = self.vpi(states).squeeze()
            return R - vpi, self.lossFunc(vpi, R.detach())

        # Baseline
        elif mode == 2:
            R = self.buffer.monte_carlo_sample(self.gamma)
            return R - R.mean(), None

    def setTarget(self, target):
        self.vpiTarget = copy.deepcopy(target)

    def resetBuffer(self):
        self.buffer = mem.ReplayBuffer(self.batch_size)
    
    def setLosses(self,act_loss,critic_loss):
        self.act_loss = act_loss
        self.critic_loss = critic_loss
        self.loss_computed = True

    def getLosses(self):
        return self.act_loss, self.critic_loss
    
    def load(self,inputDir):
        pass
    def save(self,outputDir):
        pass


class DDPG(object):
    """Deep Deterministic Policy Gradient"""

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
        self.action_dim = env.action_space.shape[0]
        self.featureExtractor = opt.featExtractor(env)
        self.state_dim = self.featureExtractor.outSize
        self.qsa_input_dim = self.state_dim + self.action_dim
        self.qsa = ut.NN(self.qsa_input_dim, 1, [dim_layers for i in range(num_layers)])
        self.optiQsa = torch.optim.Adam(params=self.qsa.parameters(),lr=1e-4)
        self.qsaTraget = copy.deepcopy(self.qsa)
        self.policy = ut.Policy_NN(self.state_dim, self.action_dim, [dim_layers for i in range(num_layers)])
        self.optiPolicy = torch.optim.Adam(params=self.policy.parameters(),lr=1e-3)
        self.policyTarget = copy.deepcopy(self.policy)
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon0 = 1
        self.mu = 10/self.nb_episodes
        self.alphal = 0
        self.alphah = 10
        self.polyakP = 0.999
        self.polyakQ = 0.999
        

    def act(self, observation, reward, frame, episode, done, succ):
        # Qtarget update
        if frame % self.targetStep == 0:
            self.setTarget(self.qsa)
        
        # Initialisation
        observation = torch.tensor(self.featureExtractor.getFeatures(observation), dtype=torch.float)
    
        # Action distribution
        distrib = Categorical(self.policy(observation))

        
        # Sample and clip action
        sample = distrib.sample() + torch.randn(2)/5
        high = torch.tensor(self.action_space.high)
        low = torch.tensor(self.action_space.low)
        a = torch.where(sample > high, high, sample)
        action = torch.where(a < low, low, a)

        if self.old_state == None:
            self.old_state = observation
            self.old_act = action
            return np.array(action)

        # Remplissage du buffer
        self.buffer.add(self.old_state, self.old_act, reward, observation, done, succ)

        # Update state and action
        self.old_state = observation
        self.old_act = action
        
        # Start training when buffer is full
        if len(self.buffer) == self.mem_size and self.test == False:
            
            self.policy.train() ; self.policyTarget.train() ; self.qsa.train() ; self.qsaTraget.train()

            states, actions, rewards, next_states, dones, succes = self.buffer.sample(self.batch_size)
            self.train_step(states, actions, rewards, next_states, dones, succes)     

            self.soft_update()

            self.policy.eval() ; self.policyTarget.eval() ; self.qsa.eval() ; self.qsaTraget.eval()

            # Smooth update of target networks params (TO DO)

        return np.array(action)
    
    def train_step(self,states, actions, rewards, next_states, dones, succes):
        """Perform a training iteration on a batch of data sampled from the experience
        replay buffer.
        """
        with torch.no_grad():
            mu_target = self.policyTarget(next_states)
            q_target = self.qsaTraget(torch.cat([next_states, mu_target], dim = 1)).squeeze()
            target = rewards + (1.0 - dones + succes) * self.gamma * q_target

        qs = self.qsa(torch.cat([states, actions], dim = 1)).squeeze()
        qsa_loss = torch.mean(self.lossFunc(qs, target.detach()))

        self.optiQsa.zero_grad()
        qsa_loss.backward()
        self.optiQsa.step()
        
        policy_loss = -torch.mean(self.qsa(torch.cat([states, self.policy(states)], dim = 1)))

        self.optiPolicy.zero_grad()
        policy_loss.backward()
        self.optiPolicy.step()
        


    def soft_update(self):

        for target, src in zip(self.policyTarget.parameters(), self.policy.parameters()):
            target.data.copy_(target.data * self.polyakP + src.data * (1-self.polyakP))

        for target, src in zip(self.qsaTraget.parameters(), self.qsa.parameters()):
            target.data.copy_(target.data * self.polyakQ + src.data * (1 - self.polyakQ))


    def setTarget(self, target):
        self.qsaTraget = copy.deepcopy(target)
    
    def save(self,outputDir):
        pass
    
    def load(self,inputDir):
        pass