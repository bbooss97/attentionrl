#this allows to test the agent given the parameters

from parallelAgent import AgentNetwork
from parallelGymEnvironment import Gymenv1player
import gym3
import argparse
#you can use the script selecting the type of network you want to test
parser=argparse.ArgumentParser()
#if more than one of these network is selected the last will be used
#if no network is selected then the feature extractore one will be used since it performs the best on average
parser.add_argument("--color", help="use color features and the position for the controller",action="store_true")
parser.add_argument("--extractedFeatures", help="use the features extracted and the position as input for the controller",action="store_true")
parser.add_argument("--position", help="use just the position for the controller",action="store_true")
parser.add_argument("--attention", help="use the attention controller",action="store_true")
parser.add_argument("--deathPenalization", help="use the network trained with a penality on deaths",action="store_true")
args=parser.parse_args()

#to do start from artifact instead of local agent

# import wandb
# run = wandb.init()
# artifact = run.use_artifact('bbooss97/attentionAgent/model:v6', type='model')
# artifact_dir = artifact.download()

# #get best parameters 
# # nameOfParameters="./parametersTesting/record.pt"
# nameOfParameters="./parameters.pt"
def testAgent(args,testing=False):
    attention=args.attention
    color=args.color
    extractedFeatures=args.extractedFeatures
    position=args.position
    deathPenalization=args.deathPenalization
    #create agent and the settings to use the network based on how it was trained
    if attention:
        color=False
        game="starpilot"
        extractorOutput=1
        qDimension=3
        kDimension=3
        useLstm=False
        useAttentionController=True
        firstBests=10
        nameOfParameters="./parametersTesting/ATTENTION CONTROLLER 8.02666666666667 game=starpilot num=20 color=False extractorOutput=1 qDimension=3 kDimension=3 useLstm=False firstBests=10 useAttentionController=True.pt"
    elif color:
        color=True
        game="starpilot"
        extractorOutput=1
        qDimension=3
        kDimension=3
        useLstm=True
        useAttentionController=False
        firstBests=10
        nameOfParameters="./parametersTesting/COLOR AND 1 EXTRACTED SHOULD BE TRAINED MORE4.757499999999993 game=starpilot num=20 color=True extractorOutput=1 qDimension=3 kDimension=3 useLstm=True firstBests=10.pt"
    elif extractedFeatures:
        color=False
        game="starpilot"
        extractorOutput=1
        qDimension=3
        kDimension=3
        useLstm=True
        useAttentionController=False
        firstBests=10
        nameOfParameters="./parametersTesting/record.pt"
    elif position:
        color=False
        game="starpilot"
        extractorOutput=0
        qDimension=4
        kDimension=4
        useLstm=True
        useAttentionController=False
        firstBests=10
        nameOfParameters="./parametersTesting/JUST THE POSITION CONTROLLER 12.27333333333334 game=starpilot num=20 color=False extractorOutput=0 qDimension=4 kDimension=4 useLstm=True firstBests=10.pt"
    elif deathPenalization:
        color=False
        game="starpilot"
        extractorOutput=1
        qDimension=3
        kDimension=3
        useLstm=True
        useAttentionController=False
        firstBests=10
        nameOfParameters="./parametersTesting/DEATHS PENALIZATION -7.133333333333326 game=starpilot num=20 color=False extractorOutput=1 qDimension=3 kDimension=3 useLstm=True firstBests=10 useAttentionController=False.pt"
    else:
        #default is using extracted features and lstm
        color=False
        game="starpilot"
        extractorOutput=1
        qDimension=3
        kDimension=3
        useLstm=True
        useAttentionController=False
        firstBests=10
        nameOfParameters="./parametersTesting/record.pt"

    #use this part just for me to test
    if testing:
        color=False
        game="starpilot"
        extractorOutput=1
        qDimension=3
        kDimension=3
        useLstm=True
        useAttentionController=False
        firstBests=10
        nameOfParameters="./parametersTesting/record.pt"


    #create the agent
    #load the agent with the settings and the weights
    agent=AgentNetwork(color=color,useLstm=useLstm,extractorOutput=extractorOutput,qDimension=qDimension,kDimension=kDimension,firstBests=firstBests,num=1,useAttentionController=useAttentionController)
    agent=agent.loadModel(nameOfParameters)
    agent.render=True
    #load parameters to network and start a game to test the agent
    agent.loadparameters(agent.getparameters())
    env=Gymenv1player(agent=agent,num=1,maxsteps=1000,verbose=False,gameName=game,render=True,blockLevel=0)
    env.env=gym3.ViewerWrapper(env.env, info_key="rgb")
    res=env.play()
    print("the final reward is:" ,res)


if __name__ == '__main__':
    testAgent(args,testing=True)