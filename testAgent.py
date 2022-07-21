from parallelAgent import AgentNetwork
from parallelGymEnvironment import Gymenv1player
import gym3


nameOfParameters="./current.pt"
gameName="starpilot"
def testAgent():
    agent=AgentNetwork(color=False,qDimension=1,kDimension=1,firstBests=10)
    agent=agent.loadModel(nameOfParameters)
    agent.render=True
    agent.loadparameters(agent.getparameters())
    env=Gymenv1player(agent=agent,num=1,maxsteps=1000,verbose=False,gameName=gameName,render=True)
    env.env=gym3.ViewerWrapper(env.env, info_key="rgb")
    res=env.play()
    print("the result is:" ,res)


if __name__ == '__main__':
    testAgent()