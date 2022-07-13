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
from agent import AgentNetwork
from agent import SelfAttention
from agent import Controller
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
    def __init__(self,gameName="coinrun",num=1,maxsteps=1000,nOfGames=1,agent=None,verbose=False):
        self.num=num
        self.maxsteps=maxsteps
        self.gameName=gameName
        self.agent=agent
        self.nOfGames=nOfGames
        self.verbose=verbose
        self.env=ProcgenGym3Env(num=num, env_name=gameName,distribution_mode="easy",use_backgrounds=False)
        
    # def play(self):
    #     reward=0
    #     step=0
    #     gamesPlayed=np.array([False for i in range(self.num)])
    #     if self.verbose:
    #         print("inizio il game")
    #     while True:
    #         if self.agent is not None:
    #             if step==0:
    #                 action=torch.tensor([4 for i in range(self.num)])
    #             else:
    #                 action=[]
    #                 for i in range(self.num):
    #                     oss=Gymenv1player.transformObs(obs,i)
    #                     # print(oss)
    #                     if gamesPlayed[i]:
    #                         a=torch.tensor([4])
    #                     else:
    #                         a=self.agent.getOutput(oss)
                        
    #                     action.append(a.squeeze())
                    
    #                 action=torch.stack(action)
                    
    #                 #print(action)
    #                 # observation=Gymenv1player.transformObs(obs)
    #                 #action=self.agent.getOutput(observation)
                
    #             self.env.act(action.to("cpu").numpy())
    #         else:
    #             action=types_np.sample(self.env.ac_space, bshape=(self.num,))
    #             self.env.act(action)
    #         rew, obs, first = self.env.observe()
            
    #         gamesPlayed=np.logical_or(gamesPlayed,first)
    #         # print(gamesPlayed,step)
    #         reward=reward+rew
    #         # print(reward)
    #         step += 1
    #         #print(action)
    #         #print(step,gamesPlayed)
    #         #print(gamesPlayed)
    #         if  step>0 and gamesPlayed.all():
    #             break
    #     if self.verbose:
    #         print("finito game")
    #     self.env.close()
    #     return reward.sum()

    def play(self):
        reward=0
        step=0
        gamesPlayed=np.array([False for i in range(self.num)])
        if self.verbose:
            print("inizio il game")
        for i in range(self.maxsteps):
            if self.agent is not None:
                if step==0:
                    a=torch.tensor([4])
                else:
                    oss=Gymenv1player.transformObs(obs,0)
                    a=self.agent.getOutput(oss)          
                a=a.to("cpu").numpy()
                self.env.act(a)
            else:
                action=types_np.sample(self.env.ac_space, bshape=(self.num,))
                self.env.act(action)
            rew, obs, first = self.env.observe()
            reward=reward+rew
            step += 1
        if self.verbose:
            print("finito game")
        self.env.close()
        return reward
    def transformObs(obs,i):
        transformedObs=np.array(obs["rgb"])
        return transformedObs[i]



if __name__ == "__main__":
    agent=AgentNetwork(qDimension=10,kDimension=10)
    agent.loadparameters([float(random.random()) for i in range(3200)])
    gymEnv=Gymenv1player(num=1,maxsteps=1000,nOfGames=1,agent=agent,gameName="starpilot")
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


