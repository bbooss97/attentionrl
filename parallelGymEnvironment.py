#this contains the parallel gym environment to create a vcectorized environment

from gym3 import types_np
from procgen import ProcgenGym3Env
import numpy as np
import torch
from parallelAgent import AgentNetwork
import random
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

class Gymenv1player():
    def __init__(self,gameName="coinrun",num=1,maxsteps=1000,agent=None,verbose=False,render=False,blockLevel=0):
        self.num=num
        #used to penalize deaths linearly
        self.lossToStayAlive=0
        #if i want to render the environment
        self.render=render
        #number of steps for every game(all of the num games )
        self.maxsteps=maxsteps
        self.gameName=gameName
        self.agent=agent
        self.verbose=verbose
        #create vectorized environment
        #num is the number of parallel games created for a single agent,
        #distriution mode easy semplifies the task otherwise the training is 8 times higher (from the procgen documentation)
        
        #block level allows to choose how many levels a game has if 0 infinites

        #if render set a render environment
        if render:
            self.env=ProcgenGym3Env(num=num, env_name=gameName,distribution_mode="easy",use_backgrounds=False,restrict_themes=True,render_mode="rgb_array")
            if blockLevel>0:
                self.env=ProcgenGym3Env(num=num, env_name=gameName,distribution_mode="easy",use_backgrounds=False,restrict_themes=True,render_mode="rgb_array",num_levels=blockLevel)
        else:
            self.env=ProcgenGym3Env(num=num, env_name=gameName,distribution_mode="easy",use_backgrounds=False,restrict_themes=True)
            if blockLevel>0:
                self.env=ProcgenGym3Env(num=num, env_name=gameName,distribution_mode="easy",use_backgrounds=False,restrict_themes=True,num_levels=blockLevel)
    #function to play a parallel environment game
    def play(self):
        #reset the reward for the num games
        reward=np.array([0 for i in range(self.num)])
        step=0
        #games played is the number of games an agent played and is increased when a player in an environment dies and restart the game
        #is used to take the averages reward of the games played for a single agent in a single game 
        gamesPlayed=np.array([1 for i in range(self.num)])
        if self.verbose:
            print("inizio il game")
        #until the max steps is reached
        for i in range(self.maxsteps):
            if self.agent is not None:
                #if its the first step nothing action 4
                if step==0:
                    #does nothing
                    a=torch.tensor([4 for i in range(self.num)])
                else:
                    #if its not the first game game the trasformed observation and pass it to the agent
                    #oss has a shape of num,64,64,3
                    oss=Gymenv1player.transformObs(obs)
                    #get action from the agent
                    a=self.agent.getOutput(oss)   
                #step in the environment given the action of the agent       
                a=a.to("cpu").numpy()
                self.env.act(a)
            else:
                #if i did not add an environment just play randomly
                action=types_np.sample(self.env.ac_space, bshape=(self.num,))
                self.env.act(action)
            #get reward next observation and if the agent died and restarted
            rew, obs, first = self.env.observe()
            #increase reward for every num games
            reward=reward+np.array(rew)
            #add games played for every num games
            dead=np.array(first).astype(np.int32)
            gamesPlayed+=dead
            dead=torch.tensor(dead)
            indexesOfDead=torch.nonzero(dead).squeeze(1)
            #reset lstm hidden state if dead
            if dead.sum(0)>0 and self.agent is not None and self.agent.useLstm:
                self.agent.controller.hidden[0][:,indexesOfDead,:]=0
                self.agent.controller.hidden[1][:,indexesOfDead,:]=0
            step += 1
        if self.verbose:
            print("finito game")
        if self.render:
            print(reward/gamesPlayed)
        
        somma=0
        #calculate the reward
        #for every parallel games(num)
        for i in range(self.num):
            #the reward of the ith game is the mean of the rewards:
            #in the game i the agent reward is the mean of the rewards of the games played
            reward[i]/=gamesPlayed[i]
            #if i selected a loss to stay alive i decrease the reward of the ith game by the loss to stay alive* the games played 
            reward[i]-=self.lossToStayAlive*gamesPlayed[i]
            #update the rewards
            somma+=reward[i]
        #return the mean of the rewards of the num games played.
        somma/=self.num
        return somma
    
    #gets the parallel observations of the vectorized environment 
    def transformObs(obs):
        transformedObs=np.array(obs["rgb"])
        # np.save("parallelObs.npy", transformedObs)
        return transformedObs

if __name__ == "__main__":
    #test the parallel environment 
    agent=AgentNetwork(num=2)
    agent.loadparameters([float(i) for i in agent.getparameters()])
    gymEnv=Gymenv1player(num=2,maxsteps=500,agent=agent,gameName="starpilot")
    print(gymEnv.play())


