import cma
import numpy as np
from parallelAgent import AgentNetwork
from parallelGymEnvironment import Gymenv1player
import tensorflow as tf
import torch
import time
# from pympler.tracker import SummaryTracker
import wandb
# import wandb
# run = wandb.init()
# artifact = run.use_artifact('bbooss97/attentionAgent/model:v6', type='model')
# artifact_dir = artifact.download()

# wandb.join()
num=20
startagain=False 
agent=AgentNetwork(color=False,qDimension=3,kDimension=3,firstBests=5,num=num)
name="cavaflyer lstm"
run=wandb.init(project='attentionAgent', entity='bbooss97',name=name)
run.watch(agent)
time.sleep(5)
def regularization(params,coeff):
    p=np.array(params)
    regularization= coeff*float(((p**2).sum())**0.5)
    return regularization

if startagain:  
    agent=agent.loadModel("./parameters.pt")
    parameters=agent.getparameters()
    parameters=[float(i) for i in parameters]
else:
    parameters=len(agent.getparameters())
    parameters=[float(0)for i in range(parameters)]


variance=1
es=cma.CMAEvolutionStrategy(parameters,variance)
j=0
whenToCopy=100

agent.cuda()

game="climber"
globalBest=-1000
start=0

# tracker = SummaryTracker()
with tf.device('/GPU:0'):
    while True:
        start+=1
        generatedParameters=es.ask()
        fitness=[]
        for i in generatedParameters:
            agent.loadparameters(i)
            env=Gymenv1player(agent=agent,maxsteps=500,verbose=False,gameName=game,num=num,blockLevel=num)
            fitness.append(100-env.play())
        es.tell(generatedParameters,fitness)
        agent.loadparameters(es.result.xfavorite)
        torch.save(agent.state_dict(), "./current.pt")
        currentBest=100-es.result.fbest
        print(currentBest)
        if currentBest>globalBest:
            print("saving current best")
            artifact = wandb.Artifact('model', type='model',)
            artifact.add_file('./current.pt')
            run.log_artifact(artifact)
            # run.finish()
            globalBest=currentBest
            agent.saveModel(str(globalBest))
        es.disp()
        run.log({"iteration":start,"globalBest":globalBest})
        # tracker.print_diff()

print(es.result.xbest)