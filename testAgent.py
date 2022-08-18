#this allows to test the agent given the parameters

from parallelAgent import AgentNetwork
from parallelGymEnvironment import Gymenv1player
import gym3
#to do start from artifact instead of local agent

# import wandb
# run = wandb.init()
# artifact = run.use_artifact('bbooss97/attentionAgent/model:v6', type='model')
# artifact_dir = artifact.download()

#get best parameters 
nameOfParameters="./parameters.pt"
gameName="caveflyer"
def testAgent():
    #create agent as the one used by the parameters
    agent=AgentNetwork(color=False,extractorOutput=1,qDimension=3,kDimension=3,firstBests=5,num=1)
    agent=agent.loadModel(nameOfParameters)
    agent.render=True
    #load parameters to network and start a game to test the agent
    agent.loadparameters(agent.getparameters())
    env=Gymenv1player(agent=agent,num=1,maxsteps=1000,verbose=False,gameName=gameName,render=True,blockLevel=0)
    env.env=gym3.ViewerWrapper(env.env, info_key="rgb")
    res=env.play()
    print("the final reward is:" ,res)


if __name__ == '__main__':
    testAgent()