import gym
import QLearner
env = gym.make('FrozenLake-v0')
ql = QLearner.QLearner(num_states=env.observation_space.n,
                       num_actions=env.action_space.n,
                       alpha=0.5,
                       rar=0.5,
                       radr=0.999,
                       gamma=0.5)
rewards_list = []
for i_episode in range(20000):
    # print ql.Q_table
    state = env.reset()
    action = ql.querysetstate(state)

    total_reward = 0
    for t in range(100):
        state, reward, done, info = env.step(action)
        action = ql.query(state, reward)
        total_reward += reward
        if done:
            # print("Episode finished after {} timesteps".format(t+1))
            # print total_reward
            rewards_list.append(total_reward)
            break
print sum(rewards_list[-1000:])