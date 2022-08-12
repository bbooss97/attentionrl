import cma
import numpy as np
from parallelAgent import AgentNetwork
from parallelGymEnvironment import Gymenv1player
# import tensorflow as tf
import torch
import os
import torch.multiprocessing as mp
import time
# from pympler.tracker import SummaryTracker
import wandb
# import wandb
# run = wandb.init()
# artifact = run.use_artifact('bbooss97/attentionAgent/model:v6', type='model')
# artifact_dir = artifact.download()
# wandb.join()
import warnings
def regularization(params,coeff):
    p=np.array(params)
    regularization= coeff*float(((p**2).sum())**0.5)
    return regularization
def train(i,shared_fitness,num,game,agent):
    # agent=AgentNetwork(qDimension=3,kDimension=3,firstBests=8,num=num)
    # agent.loadparameters(parameters)
    # agent.cuda()
    env=Gymenv1player(agent=agent,maxsteps=500,verbose=False,gameName=game,num=num,blockLevel=num)
    shared_fitness[i]=100-env.play()

if __name__ == '__main__':
    # warnings.filterwarnings("ignore")
    num=5
    num_processes = 15
    startagain=False 
    agent=AgentNetwork(qDimension=3,kDimension=3,firstBests=8,num=num)
    name="cavaflyer lstm"
    # run=wandb.init(project='attentionAgent', entity='bbooss97',name=name)
    # run.watch(agent)
    # time.sleep(5)
    agents=[]
    for i in range(num_processes):
        agents.append(AgentNetwork(qDimension=3,kDimension=3,firstBests=8,num=num).cuda())

    if startagain:  
        agent=agent.loadModel("./parameters.pt")
        parameters=agent.getparameters()
        parameters=[float(i) for i in parameters]
    else:
        parameters=len(agent.getparameters())
        parameters=[float(0)for i in range(parameters)]


    variance=1
    es=cma.CMAEvolutionStrategy(parameters,variance,inopts={
                    'popsize': num_processes,
                })
    j=0
    whenToCopy=100

    # agent.cuda()
    # agent.share_memory()
    game="climber"
    globalBest=-1000
    start=0


    shared_fitness=torch.zeros(num_processes).share_memory_()

    while True:
        start+=1
        generatedParameters=es.ask()
        fitness=[]
        processes = []
        for i in range(len(generatedParameters)):
            agents[i].loadparameters(generatedParameters[i])
            p = mp.Process(target=train, args=(i,shared_fitness,num,game,agents[i]))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print((shared_fitness-100)*-1)
        es.tell(generatedParameters,shared_fitness.tolist())
        agent.loadparameters(es.result.xfavorite)
        torch.save(agent.state_dict(), "./current.pt")
        currentBest=100-es.result.fbest
        print(currentBest)
        if currentBest>globalBest:
            print("saving current best")
            # artifact = wandb.Artifact('model', type='model',)
            # artifact.add_file('./current.pt')
            # run.log_artifact(artifact)

            globalBest=currentBest
            agent.loadparameters(es.result.xfavorite)
            agent.saveModel(str(globalBest))
        es.disp()
        # run.log({"iteration":start,"globalBest":globalBest})


    print(es.result.xbest)