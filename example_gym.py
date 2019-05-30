import gym
env = gym.make('BipedalWalkerHardcore-v2')
#env2 = gym.make('BipedalWalker-v2')
#env3 = gym.make('BipedalWalker-v2')
#env = gym.wrappers.Monitor(env, "recordingmont")
done = False
print(env.reset())
steps = 0
while not done:
    action = env.action_space.sample()
    #print(env.action_space.n)
    observation, reward, done, info = env.step(action)
    steps += 1
    #env.render()
    #print(action)
    #print(reward)
    #print(done)
    #print(info)
print(steps)
print(env.action_space)
print(env.observation_space)
env.close()
