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
        model.add(Dense(25, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def format_state(self, state):
        return np.reshape(state, [1, self.state_size])


env = gym.make('MountainCar-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
checkpoint = 500
agent = Agent(state_size, action_size, memory_size=10000, epsilon_decay=0.95)

e = 0
c = 0
while True:
    e += 1
    optimized = False
    ob = env.reset()
    max_distance = None
    total_reward = 0
    max_true_reward = 0
    for time in range(500):
        if e % 500 == 0:
            env.render()
        action = agent.act(ob)
        next_ob, reward, done, _ = env.step(action)

        total_reward += reward
        if done:
            reward = 200 + total_reward
        else:
            reward = abs(next_ob[0] - ob[0])

        max_true_reward = max(max_true_reward, reward)

        agent.remember(ob, action, reward, next_ob, done)
        ob = next_ob

        if c >= checkpoint:
            c = 0
            agent.train(batch_size)

        c += 1

        if done:
            print("episode: {}, reward: {:.2f}, e: {:.2f}"
                  .format(e, max_true_reward, agent.epsilon))
            break

