# import numpy as np
# from gym3 import types_np
# from procgen import ProcgenGym3Env
# env = ProcgenGym3Env(num=1, env_name="jumper")
# step = 0
# nOfPlays=10000
# averageReward=0
# for i in range(nOfPlays):
#     totalReward=0
#     while True:
#         #this is a random agent
        
#         env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
#         #this would be my ai
#         #env.act(np.array([0..14]))
#         rew, obs, first = env.observe()
#         totalReward+=rew      
#         #the shape of obs is 1,64,64,3 1 if squeezed
#         #print(np.array(obs["rgb"]).squeeze().shape)
#         print(f"step {step} reward {rew} first {first}")
#         if step > 0 and first:
#             # env.reset()
#             break
#         step += 1
#     averageReward+=totalReward
# averageReward/=nOfPlays
# print(averageReward)
# $ pip install procgen # install
# $ python -m procgen.interactive --env-name starpilot # human
# $ python <<EOF # random AI agent

# """
# Example random agent script using the gym3 API to demonstrate that procgen works
# """
from gym3 import types_np
from procgen import ProcgenGym3Env
import numpy as np
import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from parallelAgent import AgentNetwork

import random

# class Gymenv():
#     def __init__(self,gameName="coinrun",num=1,maxsteps=1000,agent=None):
#         self.num=num
#         self.maxsteps=maxsteps
#         self.gameName=gameName
#         self.agent=agent
#         self.env=ProcgenGym3Env(num=num, env_name=gameName)
        
#     def play(self):
#         reward=np.zeros(self.num)
#         step=0
#         for i in range(self.maxsteps):
#             if self.agent is not None:

#                 action=self.agent.getOutput(Gymenv.transformObs(obs))
#                 self.env.act(action)
#             else:
#                 self.env.act(types_np.sample(self.env.ac_space, bshape=(self.num,)))
#             rew, obs, first = self.env.observe()
#             reward=reward+rew
#             step += 1
#         averageReward=reward.sum()/self.num
#         self.env.close()
#         return averageReward
#     def transformObs(obs):
#         transformedObs=np.array(obs["rgb"]).squeeze()
#         return transformedObs


class Gymenv1player():
    def __init__(self,gameName="coinrun",num=1,maxsteps=1000,nOfGames=1,agent=None,verbose=False,render=False):
        self.num=num
        self.lossToStayAlive=0
        self.render=render
        self.maxsteps=maxsteps
        self.gameName=gameName
        self.agent=agent
        self.nOfGames=nOfGames
        self.verbose=verbose
        if render:
            self.env=ProcgenGym3Env(num=num, env_name=gameName,distribution_mode="easy",use_backgrounds=False,restrict_themes=True,render_mode="rgb_array")
        else:
            self.env=ProcgenGym3Env(num=num, env_name=gameName,distribution_mode="easy",use_backgrounds=False,restrict_themes=True)
        


    def play(self):
        reward=np.array([0 for i in range(self.num)])
        step=0
        gamesPlayed=np.array([1 for i in range(self.num)])
        if self.verbose:
            print("inizio il game")
        for i in range(self.maxsteps):
            if self.agent is not None:
                if step==0:
                    a=torch.tensor([4 for i in range(self.num)])
                else:
                    oss=Gymenv1player.transformObs(obs)
                    oss=torch.tensor(oss).cuda()
                    a=self.agent.getOutput(oss)          
                a=a.to("cpu").numpy()
                self.env.act(a)
            else:
                action=types_np.sample(self.env.ac_space, bshape=(self.num,))
                self.env.act(action)
            rew, obs, first = self.env.observe()
            reward=reward+np.array(rew)
            gamesPlayed+=np.array(first).astype(np.int32)
            step += 1
            if self.render:
                print(a)
        if self.verbose:
            print("finito game")
        if self.render:
            print(reward/gamesPlayed)
        self.env.close()
        somma=0
        for i in range(self.num):
            reward[i]/=gamesPlayed[i]
            reward[i]-=self.lossToStayAlive*gamesPlayed[i]
            somma+=reward[i]
        somma/=self.num
        return somma

    def transformObs(obs):
        transformedObs=np.array(obs["rgb"])
        # np.save("parallelObs.npy", transformedObs)
        return transformedObs

if __name__ == "__main__":
    agent=AgentNetwork(num=2,color=False)
    agent.loadparameters([float(i) for i in agent.getparameters()])

    gymEnv=Gymenv1player(num=2,maxsteps=500,nOfGames=1,agent=agent,gameName="starpilot")
#gymEnv=Gymenv1player(maxsteps=10000)
    print(gymEnv.play())

# num=10
# env = ProcgenGym3Env(num=num, env_name="coinrun")
# step = 0
# maxsteps=1000
# reward=np.zeros(num)
# for i in range(maxsteps):
#     env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
#     rew, obs, first = env.observe()
#     reward=reward+rew
#     step += 1
# print(reward)
# print(reward.sum()/num)
# env.close()


