import cma
import numpy as np
from agent import AgentNetwork
from gymEnvironment import Gymenv1player
# optimizer=cma.CMAEvolutionStrategy(x0=[100,100],sigma0=1)
# def function(x):
#     return x[0]**2+x[1]**2

# optimizer.optimize(objective_fct=function,verb_disp=1)
# print(optimizer.result.xbest)

# >>> es = cma.CMAEvolutionStrategy(4 * [1], 1)  # doctest: +ELLIPSIS
# (4_w,8)-aCMA-ES (mu_w=2.6,w_1=52%) in dimension 4 (seed=...
# >>> while not es.stop():
# ...    X = es.ask()
# ...    es.tell(X, [cma.ff.elli(x) for x in X])
# ...    es.disp()  # doctest: +ELLIPSIS

parameters=AgentNetwork().getparameters()
variance=1
es=cma.CMAEvolutionStrategy(parameters,variance)
j=0
agent=AgentNetwork()
while True:
    generatedParameters=es.ask()
    fitness=[]
    for i in generatedParameters:
        agent.loadparameters(i)
        env=Gymenv1player(agent=agent,nOfGames=1,maxsteps=250)
        fitness.append(1000-env.play())
    es.tell(generatedParameters,fitness)
    es.disp()
    if j%1000==999:
        agent=AgentNetwork()
        agent.loadparameters(es.result.xbest)
        agent.saveModel()
        print(Gymenv1player(agent=agent).play())
    j+=1
print(es.result.xbest)