from agent import AgentNetwork
from gymEnvironment import Gymenv1player
import gym3

nameOfParameters="./params/28.pt"

def testAgent():
    agent=AgentNetwork.loadModel(nameOfParameters)
    agent.loadparameters(agent.getparameters())
    env=Gymenv1player(agent=agent,num=1,maxsteps=1000,verbose=False,gameName="starpilot",render=True)
    env.env=gym3.ViewerWrapper(env.env, info_key="rgb")
    res=env.play()
    print(res)


if __name__ == '__main__':
    testAgent()