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

    n_frame_history = 4

    def _create_model(self):
        # create model
        model = Sequential()
        # normalize input
        model.add(Lambda(lambda x: x / 255.0, input_shape=(105, 80, self.n_frame_history, )))
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
        # downsample each frame
        down_sample = np.array([im.preprocess(frame) for frame in state])
        state = np.stack(down_sample, axis=2)

        return state


env = gym.make('Breakout-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
target_frame = 1000
max_frames = 50000
prefill_size = 20000
n_frame_history = 4
frame_history = []

real_mode = len(sys.argv) > 1 and sys.argv[1] == 'real'
fast_mode = len(sys.argv) > 1 and sys.argv[1] == 'fast'
if real_mode:
    agent = Agent(state_size, action_size,
                  epsilon_decay_policy=[0.1],
                  model_path='./.models/breakout.h5',)
else:
    agent = Agent(state_size, action_size,
                  memory_size=max_frames,
                  prefill_size=prefill_size,
                  epsilon_decay_policy=[1.0, 0.1],
                  model_path='./.models/breakout.h5',
                  gamma=0.99,
                  learning_rate=0.0001)

agent.model.summary()

frames = 0
while True:

    optimized = False

    ob = env.reset()
    # copy screens 4 times
    frame_history = 4 * [ob]

    max_distance = None
    total_reward = 0
    lives = 5

    env.step(1)
    while True:
        if not fast_mode:
            env.render()
        action = agent.act(frame_history)
        next_ob, reward, done, info = env.step(action)
        next_frame_history = frame_history[1:]  # push and pop history
        next_frame_history.append(next_ob)
        total_reward += reward

        # subtract score if you lose a life
        if lives != info.get('ale.lives'):
            lives = info.get('ale.lives')
            reward -= 1

        agent.remember(frame_history, action, reward, next_frame_history, done)
        frame_history = next_frame_history


        if frames > target_frame and not real_mode:
            sys.stdout.write(u"\u001b[1000Depisode: {}, average reward: {:.5f}, e: {:.1f}, m: {}..."
                             .format(agent.training_sessions, agent.get_average_reward(), agent.epsilon, len(agent.session)))
            sys.stdout.flush()

            if not real_mode:
                if agent.train(batch_size):
                    sys.stdout.write("saving...")
                    agent.save('./.models/breakout.h5')
                    sys.stdout.write("OK\n")
                else:
                    sys.stdout.write("waiting for prefill\n")
                sys.stdout.flush()
            frames = 0
        else:
            sys.stdout.write(u"\u001b[1000DPlaying...{:.2f}%".format(frames / target_frame * 100))
            sys.stdout.flush()


        if done:
            break

        frames += 1

        if real_mode:
            sleep(1/60)
