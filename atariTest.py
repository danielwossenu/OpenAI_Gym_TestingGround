import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    print i_episode
    for t in range(10000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print env.action_space.n
        if reward != 0.0:
            print reward
            pass
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
