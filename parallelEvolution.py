import cma
import numpy as np
from parallelAgent import AgentNetwork
from parallelGymEnvironment import Gymenv1player
import tensorflow as tf
import torch
# from pympler.tracker import SummaryTracker

def regularization(params,coeff):
    p=np.array(params)
    regularization= coeff*float(((p**2).sum())**0.5)
    return regularization
num=10
parameters=AgentNetwork().getparameters()


variance=1
es=cma.CMAEvolutionStrategy(parameters,variance)
j=0
whenToCopy=100

agent=AgentNetwork(color=False,qDimension=3,kDimension=3,firstBests=10,num=num)
agent.cuda()
game="starpilot"
globalBest=0
# tracker = SummaryTracker()
with tf.device('/GPU:0'):
    while True:
        generatedParameters=es.ask()
        fitness=[]
        for i in generatedParameters:
            agent.loadparameters(i)
            env=Gymenv1player(agent=agent,maxsteps=500,verbose=False,gameName=game,num=num)
            fitness.append(100-env.play())
        es.tell(generatedParameters,fitness)
        agent.loadparameters(es.result.xbest)
        env=Gymenv1player(agent=agent,maxsteps=500,verbose=False,gameName=game,num=num)
        currentBest=env.play()
        torch.save(agent.state_dict(), "./current.pt")
        print(currentBest)
        if currentBest>=globalBest:
            print("saving current best")
            globalBest=currentBest
            agent.saveModel(str(globalBest))
        es.disp()
        
        # tracker.print_diff()

print(es.result.xbest)