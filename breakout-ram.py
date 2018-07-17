#!/usr/bin/env python3

import gym
import numpy as np
import sys
from time import sleep

from Agent import Agent as BaseAgent
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense


class Agent(BaseAgent):

    def _create_model(self):
        # create model
        model = Sequential()
        model.add(Dense(512, input_shape=(self.state_size,), activation='relu', kernel_initializer='zero',
                        bias_initializer='zero'))
        model.add(Dense(256, activation='relu', kernel_initializer='zero', bias_initializer='zero'))
        model.add(Dense(128, activation='relu', kernel_initializer='zero', bias_initializer='zero'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='zero', bias_initializer='zero'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def format_state(self, state):
        return np.reshape(state, [1, self.state_size])


env = gym.make('Breakout-ram-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32

real_mode = len(sys.argv) > 1 and sys.argv[1] == 'real'
fast_mode = len(sys.argv) > 1 and sys.argv[1] == 'fast'
if real_mode:
    agent = Agent(state_size, action_size, epsilon=-1.0, model_path='./.models/breakout-ram.h5')
else:
    agent = Agent(state_size, action_size,
                  memory_size=10000, epsilon_decay=0.95, model_path='./.models/breakout-ram.h5')

e = 0
while True:
    e += 1
    optimized = False

    ob = env.reset()
    max_distance = None
    total_reward = 0
    life = None
    t = 0
    env.step(1)
    while True:
        if not fast_mode:
            env.render()
        action = agent.act(ob)
        next_ob, reward, done, info = env.step(action)
        if info.get('ale.lives') != 5:
            reward = 0
        total_reward += reward

        agent.remember(ob, action, reward, next_ob, done)
        ob = next_ob

        if done or t > 500 or info.get('ale.lives') != 5:
            sys.stdout.write("episode: {}, reward: {:.2f}, e: {:.2f} , t: {}..."
                             .format(e, total_reward, agent.epsilon, t))
            sys.stdout.flush()

            if not real_mode:
                agent.train(batch_size)

            sys.stdout.write("OK\n")
            sys.stdout.flush()
            t = 0

        if done or info.get('ale.lives') != 5:
            break

        t += 1

        if real_mode:
            sleep(1/60)

    if e % 10 == 0 and not real_mode:
        sys.stdout.write("saving...")
        agent.save('./.models/breakout-ram.h5')
        sys.stdout.write("OK\n")