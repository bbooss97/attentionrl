# import gym
# env = gym.make("LunarLander-v2")
# observation, info = env.reset(seed=42, return_info=True)
# for _ in range(1000):
#    env.render()
#    action = policy(observation)  # User-defined policy function
#    observation, reward, done, info = env.step(action)

#    if done:
#       observation, info = env.reset(return_info=True)
# env.close()




# import gym
# envs = gym.envs.registry.all()
# print(envs)
# print('Total envs available:', len(envs))

# import gym
# env = gym.make('MountainCar-v0')
# # Uncomment following line to save video of our Agent interacting in this environment
# # This can be used for debugging and studying how our agent is performing
# # env = gym.wrappers.Monitor(env, './video/', force = True)
# t = 0
# while True:
#    t += 1
#    env.render()
#    observation = env.reset()
#    print(observation)
#    action = env.action_space.sample()
#    observation, reward, done, info = env.step(action)
#    if done:
#       print("Episode finished after {} timesteps".format(t+1))
#    break
# env.close()

# $ pip install procgen # install
# $ python -m procgen.interactive --env-name starpilot # human
# $ python <<EOF # random AI agent


from gym3 import types_np
from procgen import ProcgenGym3Env
import numpy as np
num=100
env = ProcgenGym3Env(num=num, env_name="coinrun")
step = 0
reward=np.zeros(num)
for i in range(1000):
    env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
    rew, obs, first = env.observe()
    reward=reward+rew
    #print(f"step {step} reward {rew} first {first}")
    step += 1
print(reward)
print(reward.sum()/num)

"""
Example random agent script using the gym3 API to demonstrate that procgen works
"""
# import numpy as np
# from gym3 import types_np
# from procgen import ProcgenGym3Env
# env = ProcgenGym3Env(num=1, env_name="jumper")
# step = 0
# nOfPlays=10000
# averageReward=0
# for i in range(nOfPlays):
#     totalReward=0
#     while True:
#         #this is a random agent
        
#         env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
#         #this would be my ai
#         #env.act(np.array([0..14]))
#         rew, obs, first = env.observe()
#         totalReward+=rew      
#         #the shape of obs is 1,64,64,3 1 if squeezed
#         #print(np.array(obs["rgb"]).squeeze().shape)
#         print(f"step {step} reward {rew} first {first}")
#         if step > 0 and first:
#             # env.reset()
#             break
#         step += 1
#     averageReward+=totalReward
# averageReward/=nOfPlays
# print(averageReward)