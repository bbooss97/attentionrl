import cma
import numpy as np
from agent import AgentNetwork
from gymEnvironment import Gymenv1player
# from pympler.tracker import SummaryTracker

parameters=AgentNetwork().getparameters()
parameters=[float(0) for i in range(3200)]

variance=0.1
es=cma.CMAEvolutionStrategy(parameters,variance,inopts={'popsize':50})
j=0
whenToCopy=100
agent=AgentNetwork()
agent.cuda()
game="starpilot"
globalBest=0
# tracker = SummaryTracker()
while not es.stop():
    generatedParameters=es.ask()
    fitness=[]
    for i in generatedParameters:
        agent.loadparameters(i)
        env=Gymenv1player(agent=agent,num=1,maxsteps=1000,verbose=False,gameName=game)
        fitness.append(100-env.play())
    es.tell(generatedParameters,fitness)
    agent.loadparameters(es.result.xbest)
    env=Gymenv1player(agent=agent,num=1,maxsteps=1000,verbose=False,gameName=game)
    currentBest=env.play()
    print(currentBest)
    if currentBest>globalBest:
        print("saving current best")
        globalBest=currentBest
        agent.saveModel()
    es.disp()
    
    # tracker.print_diff()

print(es.result.xbest)