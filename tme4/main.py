import copy
import numpy as np
from collections import defaultdict
import utils as ut
import memory as mem
import argparse
import sys
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use("Qt5agg")
#matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from torch.utils.tensorboard import SummaryWriter
import gridworld
import torch
from utils import *
from agents import DQNAgent, PolicyGradAgent



class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env, opt):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def save(self,outputDir):
        pass

    def load(self,inputDir):
        pass


def train(batch_size, target_step, dim_layers, num_layers, lr1, lr2, loss_num, mu, log=False, verb=False):
    #config = load_yaml('./configs/config_random_gridworld.yaml')
    config = load_yaml('./configs/config_random_cartpole.yaml')
    #config = load_yaml('./configs/config_random_lunar.yaml')

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]

    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/" + config["env"] + "/Grid_actor_critic/" + "_bs" + str(batch_size) + "_dim" +str(dim_layers) +"_num" +str(num_layers)+"_lr1"+str(lr1)+"_lr2"+str(lr2)+"_loss"+str(loss_num)+tstart


    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"]
    ob = env.reset()

    agent_0 = RandomAgent(env,config)
    agent_1 = DQNAgent(env,config,episode_count, batch_size, target_step, dim_layers, num_layers)
    agent_2 = PolicyGradAgent(env,config,episode_count, batch_size, target_step, dim_layers, num_layers, lr1, lr2, mu,loss_Func = loss_num)

    agent = agent_2

    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    if log:
        logger = LogMe(SummaryWriter(outdir))
        loadTensorBoard(outdir)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    cur_frame = 0
    train_reward = []
    test_reward = []
    for i in range(episode_count):
        if i % int(config["freqVerbose"]) == 0 and i >= config["freqVerbose"]:
            verbose = verb
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            #print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            #print("End of test, mean reward=", mean / nbTest)
            itest += 1
            if log:
                logger.direct_write("rewardTest", mean / nbTest, itest)
            test_reward.append(mean / nbTest)
            agent.test = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        while True:
            if verbose:
                env.render()

            action = agent.act(ob, reward, cur_frame, i, done)
            ob, reward, done, _ = env.step(action)
            cur_frame+=1
            j+=1

            rsum += reward
            if done:
                #print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                if log:
                    logger.direct_write("reward", rsum, i)
                train_reward.append(rsum)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                ob = env.reset()
                break

    env.close()
    train_rg = np.arange(len(train_reward))
    test_rg = np.arange(len(test_reward))
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].plot(train_rg,train_reward)
    ax[1].plot(test_rg,test_reward)
    fig.savefig(outdir+ "/reward.png")
    plt.close(fig)

    return sum(test_reward)



# Grid Search
GRID = False

if GRID:
    batch_l = [20,50,100,200,500]
    dim_layers = [32,64,128]
    num_layers = [1,2,3]
    lr1, lr2 = [1e-3, 1e-4], [1e-3,1e-4]
    #mu = [20,30,40]
    loss = [0,1]
    out = open("out.txt", "a")
    max_reward = 0
    for b in batch_l:
        for d in dim_layers:
            for n in num_layers:
                for l1 in lr1:
                    for l2 in lr2:
                        for l in loss:
                            #d,n,l1,l2 = 32,1,1e-3,1e-3
                            m = 10 # MSELoss
                            t=1
                            print(f"batch{b}_dim{d}_num{n}_l1{l1}_l2{l2}_l{l}\n")
                            rcum = train(b, t, d, n, l1, l2, l,m,log=False,verb=False)
                            if rcum > max_reward:
                                max_reward = rcum
                                best_params = {"batch" : b, "dim":d,"num":n,"l1":l1,"l2": l2,"mu":m}
                                
                            print("-"*20)
                            print("best_params\n", best_params)
                            print("max reward cum :", max_reward)

    out.write("Best params :",best_params)
    out.close()

else:
    # Hyper param

    batch_size = 100
    target_step = 1 # Pas utile pour actor critic -> changement de target à chaque optim
    dim_layers = 32
    num_layers = 2
    lr1, lr2 = 1e-4, 1e-3
    loss_num = 1
    mu = 30

    rcum = train(batch_size, target_step, dim_layers, num_layers, lr1, lr2, loss_num, mu,log=True,verb=False)
    print("-"*20)
    print(f"\nreward cum :{rcum}")