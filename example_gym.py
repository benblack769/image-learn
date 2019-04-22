import gym
env = gym.make('Pong-v0')
#env = gym.wrappers.Monitor(env, "recording")
done = False
print(env.reset())
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    #env.render()
    print(action)
    print(reward)
    print(done)
    print(info)
print(env.action_space)
env.close()
