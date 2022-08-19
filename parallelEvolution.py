#this is the script to train parallel agents 

import cma
import numpy as np
from parallelAgent import AgentNetwork
from parallelGymEnvironment import Gymenv1player
import torch
import time
import wandb
import pickle

# run = wandb.init()
# artifact = run.use_artifact('bbooss97/attentionAgent/model:v6', type='model')
# artifact_dir = artifact.download()

#dump cmaes execution to load from it if i want to continue
filename = './outcmaes/es-pickle-dump'
#number of games an agent plays in a parallel way (batched execution)

#continue training from a previous execution
startagain=True 
#agent with his parameters look the parallel agent file
num=10
game="starpilot"
color=False
extractorOutput=1
qDimension=3
kDimension=3
useLstm=True
firstBests=10
agent=AgentNetwork(color=color,useLstm=useLstm,extractorOutput=extractorOutput,qDimension=qDimension,kDimension=kDimension,firstBests=firstBests,num=num)
#wandb run
name="game={} num={} color={} extractorOutput={} qDimension={} kDimension={} useLstm={} firstBests={}".format(game, num, color, extractorOutput, qDimension, kDimension, useLstm, firstBests)

run=wandb.init(project='attentionAgent', entity='bbooss97',name=name)
run.watch(agent)

time.sleep(5)
#load parameters from the previous best else from scratch
if startagain:  
    agent=agent.loadModel("./parameters.pt")
    parameters=agent.getparameters()
    parameters=[float(i) for i in parameters]
else:
    parameters=len(agent.getparameters())
    parameters=[float(0)for i in range(parameters)]
# initial variance of cmaes algorithm
variance=1
#load previous training if i want to continue from it
if startagain:
   es = pickle.load(open(filename, 'rb'))
else:
    es=cma.CMAEvolutionStrategy(parameters,variance)
#put the agent to cuda so that i can evaluate the num games in parallel
agent.cuda()

globalBest=-1000
start=0

#strat training
while True:
    start+=1
    #generate parameters from cmaes
    generatedParameters=es.ask()
    fitness=[]
    #for every parameters play num games in parallel and take the average raward of the num games
    for i in generatedParameters:
        #load the parameters to the agent
        agent.loadparameters(i)
        #create vectorized environment and get fitnesses
        env=Gymenv1player(agent=agent,maxsteps=500,verbose=False,gameName=game,num=num,blockLevel=0)
        fitness.append(100-env.play())
    #print mean of all the generated parameters executions
    mean=100-torch.tensor(fitness).mean()
    print("mean: {}".format(mean))
    #send fitnesses to cmaes and load best parameters to save the current agent
    es.tell(generatedParameters,fitness)
    agent.loadparameters(es.result.xfavorite)
    torch.save(agent.state_dict(), "./current.pt")
    currentBest=100-es.result.fbest
    print(currentBest)
    #if i have a record i save the parameters locally and on wandb
    if currentBest>globalBest:
        print("saving current best")
        artifact = wandb.Artifact('model', type='model',)
        artifact.add_file('./current.pt')
        run.log_artifact(artifact)
        globalBest=currentBest
        agent.saveModel("parametersTraining/"+str(globalBest)+" "+name)
    #log to wandb and save the execution of cmaes in case i want to continue later
    es.disp()
    run.log({"iteration":start,"globalBest":globalBest," mean":mean})
    open(filename, 'wb').write(es.pickle_dumps())

