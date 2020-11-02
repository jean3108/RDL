import gym
import gridworld
import copy
import numpy as np
from collections import defaultdict
import utils as ut
import memory as mem
import torch



class DQNAgent(object):
    """Deep Qlearning"""

    def __init__(self, env, opt, test = False):

        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)

        self.lossFunc = torch.nn.SmoothL1Loss()
        self.targetStep = 1000
        self.batch_size = 100
        self.mem_size = 10000
        self.buffer = mem.Memory(self.mem_size)
        self.test = test
        self.old_state = None
        self.old_act = None
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.Qsa = ut.NN(self.featureExtractor.outSize, env.action_space.n)
        self.QsaTraget = self.Qsa
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon0 = 0.2
        self.mu = 0.2

    def act(self, observation, reward, time, done):

        # Qtarget update
        if time % self.targetStep == 0:
            self.setTarget(self.Qsa) 
        
        # Initialisation
        observation = torch.tensor(self.featureExtractor.getFeatures(observation), dtype=torch.float)
        qs = self.Qsa(observation)

        if self.old_state == None:
            
            action = self.env.action_space.sample()
            self.old_state = observation
            self.old_act = action
            
            return action

        # epsilon greedy
        eps = self.epsilon0 / (1 + self.mu * time)
        if np.random.rand() > eps or self.test == True:
            action = torch.argmax(qs)

        else:
            action = self.env.action_space.sample()

        # Remplissage du buffer
        transition = [self.old_state, self.old_act, observation, reward, done]
        self.buffer.store(transition)

        # Apprentissage
        if self.buffer.mem_ptr == self.buffer.mem_size:
            samples = self.buffer.sample(self.batch_size)
            X = torch.tensor([self.Qsa(sample[0])[sample[1]] for sample in samples])
            with torch.no_grad():
                Y = torch.tensor([sample[3] if sample[4] == True else sample[3] + self.gamma * np.max(self.QsaTraget(sample[2])) for sample in samples])

            loss = self.lossFunc(X, Y)
            loss.backward()

        # Update state and action
        self.old_state = observation
        self.old_act = action
            
        return action

    def setTarget(self, target):
        self.QsaTraget = copy.deepcopy(target)
    
    def save(self,outputDir):
        pass
    
    def load(self,inputDir):
        pass



class PolicyGradAgent(object):
    """Deep Qlearning"""

    def __init__(self, env, opt, test = False):

        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)

        self.lossFunc = torch.nn.SmoothL1Loss()
        self.targetStep = 1000
        self.batch_size = 100
        self.mem_size = 10000
        self.buffer = mem.Memory(self.mem_size)
        self.test = test
        self.old_state = None
        self.old_act = None
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.Vpi = ut.NN(self.featureExtractor.outSize, 1) # V value to estimate Actor
        self.Policy = ut.NN(self.featureExtractor.outSize, env.action_space.n) # Policy 
        self.VpiTarget = self.Vpi
        self.OptiVpi = torch.optim.Adam(params=self.Vpi.parameters(),lr=1e-3)
        self.OptiPolicy = torch.optim.Adam(params=self.Policy.parameters(),lr=1e-3)
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon0 = 0.2
        self.mu = 0.2

    def act(self, observation, reward, time, done):

        # Qtarget update
        if time % self.targetStep == 0:
            self.setTarget(self.Vpi) 
        
        # Initialisation
        observation = torch.tensor(self.featureExtractor.getFeatures(observation), dtype=torch.float)
        qs = self.Policy(observation)

        if self.old_state == None:
            
            action = self.env.action_space.sample()
            self.old_state = observation
            self.old_act = action
            
            return action

        # epsilon greedy
        eps = self.epsilon0 / (1 + self.mu * time)
        if np.random.rand() > eps or self.test == True:
            action = torch.argmax(qs)

        else:
            action = self.env.action_space.sample()

        # Remplissage du buffer
        transition = [self.old_state, self.old_act, observation, reward, done]
        self.buffer.store(transition)

        # Apprentissage des Vpi
        if self.buffer.mem_ptr == self.buffer.mem_size:

            self.OptiVpi.zero_grad() # clear Vpi params gradient
            self.OptiPolicy.zero_grad() # clear Policy params gradient
            samples = self.buffer.sample(self.batch_size)
            X_vpi = torch.tensor([self.Vpi(sample[0]) for sample in samples])
            with torch.no_grad():
                Y_vpi = torch.tensor([sample[3] if sample[4] == True else sample[3] + self.gamma * self.Vpi(sample[2]) for sample in samples])
                
            loss = self.lossFunc(X_vpi, Y_vpi)
            loss.backward()
            self.OptiVpi.step()

            # Campute Actor
            Actor = torch.tensor([sample[3] + self.gamma*self.Vpi(sample[2]) - self.Vpi(sample[0]) for sample in samples])


            # Compute reward expectation gradient
            # 1) compute expectation
            logProba = torch.tensor([torch.log(self.Policy(sample[0])[sample[1]]) for sample in samples])
            expect = torch.sum(logProba*Actor)

            # 2) back propagation
            expect.backward()

            # 3) Update parameters
            self.OptiPolicy.step()

        # Update state and action
        self.old_state = observation
        self.old_act = action
            
        return action

    def setTarget(self, target):
        self.VpiTarget = copy.deepcopy(target)
    
    def save(self,outputDir):
        pass
    
    def load(self,inputDir):
        pass