#!/usr/bin/env python3

import gym
import numpy as np

from Agent import Agent as BaseAgent
from collections import deque
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


env = gym.make('MountainCar-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
agent = Agent(state_size, action_size)


success_stream = deque(maxlen=3)
e = 0
while True:
    e += 1
    optimized = False
    ob = env.reset()
    max_distance = None
    for time in range(500):
        # env.render()
        action = agent.act(ob)
        next_ob, reward, done, _ = env.step(action)
        agent.remember(ob, action, reward, next_ob, done)
        ob = next_ob

        if max_distance is None:
            max_distance = ob[0]
        else:
            max_distance = max(max_distance, ob[0])

        if ob[0] >= 0.5:
            optimized = True
            break

        if done:
            if optimized:
                success_stream.append(100)
            else:
                success_stream.append(0)
            goal_percent = (max_distance+1.2) / (0.6+1.2) * 100
            print("episode: {}, goal: {:.2f}%, e: {:.2f} success rate: {:.2f}"
                  .format(e, goal_percent, agent.epsilon, np.average(success_stream)))
            break

    agent.train(batch_size)

    # finish if success rate is 9 out of 10
    if np.average(success_stream) > 90:
        break


ob = env.reset()
while True:
    env.render()
    action = agent.act(ob)
    ob, reward, done, _ = env.step(action)
    if done:
        ob = env.reset()
