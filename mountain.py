#!/usr/bin/env python3

import gym
import numpy as np

from Agent import Agent


EPISODES = 10000


env = gym.make('MountainCar-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
agent = Agent(state_size, action_size)

optimized = False
for e in range(EPISODES):
    state = np.reshape(env.reset(), [1, state_size])
    max_distance = None
    for time in range(500):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if max_distance is None:
            max_distance = state[0][0]
        else:
            max_distance = max(max_distance, state[0][0])

        if state[0][0] >= 0.5:
            optimized = True
            break

        if done:
            print("episode: {}/{}, distance from goal: {}, e: {:.2}".format(e, EPISODES, abs(0.5-max_distance), agent.epsilon))
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
    if done:
        state = np.reshape(env.reset(), [1, state_size])
