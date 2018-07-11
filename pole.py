#!/usr/bin/env python3

import gym
import numpy as np

from Agent import Agent


EPISODES = 10000


env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
agent = Agent(state_size, action_size)

optimized = False
for e in range(EPISODES):
    state = np.reshape(env.reset(), [1, state_size])
    for time in range(500):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if time >= 499:
            optimized = True
            break

        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
            break
    if optimized:
        break
    agent.train(batch_size)


state = np.reshape(env.reset(), [1, state_size])
while True:
    env.render()
    action = agent.act(state)
    ob, reward, done, _ = env.step(action)
    state = np.reshape(ob, [1, state_size])
