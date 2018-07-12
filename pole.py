#!/usr/bin/env python3

import gym
import numpy as np

from Agent import Agent as BaseAgent
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense


class Agent(BaseAgent):

    def _create_model(self):
        # create model
        model = Sequential()
        model.add(Dense(25, input_shape=(self.state_size, ), activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def format_state(self, state):
        return np.reshape(state, [1, self.state_size])


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
    if done:
        state = np.reshape(env.reset(), [1, state_size])
