#!/usr/bin/env python3

import gym
import numpy as np
import sys

from Agent import Agent as BaseAgent
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Flatten


class Agent(BaseAgent):

    def _create_model(self):
        # create model
        model = Sequential()
        model.add(Conv2D(4, 84, activation='relu', input_shape=(210, 160, 3,)))
        model.add(Conv2D(16, 20, activation='relu'))
        model.add(Conv2D(32, 9, activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def format_state(self, state):
        return np.expand_dims(state, 0)


env = gym.make('Breakout-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
agent = Agent(state_size, action_size, memory_size=10000, epsilon_decay=0.95)
agent.model.summary()
# sys.exit(0)
e = 0
while True:
    e += 1
    optimized = False

    ob = env.reset()
    max_distance = None
    total_reward = 0
    while True:
        if not (len(sys.argv) > 1 and sys.argv[1] == 'fast'):
            env.render()
        action = agent.act(ob)
        next_ob, reward, done, _ = env.step(action)

        total_reward += reward

        agent.remember(ob, action, reward, next_ob, done)
        ob = next_ob

        if done:
            sys.stdout.write("episode: {}, reward: {:.2f}, e: {:.2f}..."
                             .format(e, total_reward, agent.epsilon))
            sys.stdout.flush()
            agent.train(batch_size)
            sys.stdout.write("OK\n")
            sys.stdout.flush()
            break

    if e % 10 == 0:
        sys.stdout.write("saving...")
        # agent.save('./.models/breakout-ram.h5')
        sys.stdout.write("OK\n")