from parallelAgent import AgentNetwork
from parallelGymEnvironment import Gymenv1player
import gym3


nameOfParameters="./parameters.pt"
gameName="coinrun"
def testAgent():
    agent=AgentNetwork(color=False,qDimension=3,kDimension=3,firstBests=10)
    agent=agent.loadModel(nameOfParameters)
    agent.render=True
    agent.loadparameters(agent.getparameters())
    env=Gymenv1player(agent=agent,num=1,maxsteps=1000,verbose=False,gameName=gameName,render=True)
    env.env=gym3.ViewerWrapper(env.env, info_key="rgb")
    res=env.play()
    print("the result is:" ,res)


if __name__ == '__main__':
    testAgent()