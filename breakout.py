#!/usr/bin/env python3

import gym
import numpy as np
import sys
from time import sleep
import imageprocessing as im
from Agent import Agent as BaseAgent
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Flatten, Lambda


class Agent(BaseAgent):

    def _create_model(self):
        # create model
        model = Sequential()
        # normalize input
        model.add(Lambda(lambda x: x / 255.0, input_shape=(105, 80, 1, )))
        # first layer
        model.add(Conv2D(16, 8, activation='relu'))
        # second layer
        model.add(Conv2D(32, 4, activation='relu'))
        # flattened
        model.add(Flatten())
        # first hidden
        model.add(Dense(256, activation='linear'))
        # second hidden
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def format_state(self, state):
        state = np.reshape(im.preprocess(state), (105, 80, 1))
        return state


env = gym.make('Breakout-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
target_frame = 500# 40000
max_frames = 80000000

real_mode = len(sys.argv) > 1 and sys.argv[1] == 'real'
fast_mode = len(sys.argv) > 1 and sys.argv[1] == 'fast'
if real_mode:
    agent = Agent(state_size, action_size, model_path='./.models/breakout.h5')
else:
    agent = Agent(state_size, action_size,
                  memory_size=max_frames,
                  epsilon_decay_policy=[1.0, 0.1],
                  model_path='./.models/breakout.h5',
                  gamma=0.99,
                  learning_rate=0.0001)

agent.model.summary()

frames = 0
while True:

    optimized = False

    ob = env.reset()
    max_distance = None
    total_reward = 0
    life = None

    env.step(1)
    while True:
        if not fast_mode:
            env.render()
        action = agent.act(ob)
        next_ob, reward, done, info = env.step(action)
        total_reward += reward

        agent.remember(ob, action, reward, next_ob, done)
        ob = next_ob

        if frames > target_frame and not real_mode:
            sys.stdout.write("episode: {}, average reward: {:.10f}, e: {:.4f}..."
                             .format(agent.training_sessions, agent.get_average_reward(), agent.epsilon))
            sys.stdout.flush()

            if not real_mode:
                agent.train(batch_size)

            sys.stdout.write("OK\n")
            sys.stdout.flush()

            sys.stdout.write("saving...")
            agent.save('./.models/breakout.h5')
            sys.stdout.write("OK\n")
            frames = 0

        if done:
            break

        frames += 1

        if real_mode:
            sleep(1/60)
