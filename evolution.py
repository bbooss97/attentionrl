import cma
import numpy as np
from agent import AgentNetwork
from gymEnvironment import Gymenv1player

parameters=AgentNetwork().getparameters()
parameters=[float(0) for i in range(3200)]

variance=0.1
es=cma.CMAEvolutionStrategy(parameters,variance,inopts={'popsize':50})
j=0
whenToCopy=100
agent=AgentNetwork()
game="starpilot"
while not es.stop():
    generatedParameters=es.ask()
    fitness=[]
    for i in generatedParameters:
        agent.loadparameters(i)
        env=Gymenv1player(agent=agent,num=5,maxsteps=1000,verbose=False,gameName=game)
        fitness.append(100-env.play())
    es.tell(generatedParameters,fitness)
    es.disp()
    if j%whenToCopy==whenToCopy-1:
        agent.loadparameters(es.result.xbest)
        agent.saveModel()
        print(Gymenv1player(agent=agent).play())
    j+=1
print(es.result.xbest)