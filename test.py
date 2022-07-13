from re import A
import numpy as np
import cma
from agent import *


import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

pool = mp.Pool(mp.cpu_count())



def provaEvoluzione():
    parameters=np.array([0 for i in range(1000)])
    def funzione(parameters):
        return (parameters**2).sum()
    variance=1
    es=cma.CMAEvolutionStrategy(parameters,variance,{'popsize':200})
    j=0
    while not es.stop():
        generatedParameters=np.array(es.ask())
        fitness=(generatedParameters**2).sum(axis=1)
       
        es.tell(generatedParameters,fitness)
        es.disp()
        if j%1000==999:
            print(es.result.xbest)
        j+=1

def provaAgente():
    agent=AgentNetwork()

#provaEvoluzione()
