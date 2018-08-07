#!/usr/bin/env python3
import numpy as np
import gym
import sys
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
import imageprocessing as im


ENV_NAME = 'Breakout-v0'
WEIGHTS_PATH = '.models/dqn_{}_weights.h5f'.format(ENV_NAME)
INPUT_SHAPE = (105, 80)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = im.preprocess(observation)
        assert img.shape == INPUT_SHAPE
        return img

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)



# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
model.add(Permute((2, 3, 1), input_shape=input_shape))
model.add(Conv2D(32, 8, activation='relu'))
model.add(Conv2D(64, 4, activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='relu'))
model.summary()


train_mode = len(sys.argv) > 1 and sys.argv[1] == 'train'


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=20000, window_length=WINDOW_LENGTH)
policy = BoltzmannQPolicy()
processor = AtariProcessor()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, enable_dueling_network=True,
               dueling_type='avg', processor=processor)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

if os.path.isfile(WEIGHTS_PATH) and os.access(WEIGHTS_PATH, os.R_OK):
    dqn.load_weights(WEIGHTS_PATH)

if train_mode:

    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    dqn.save_weights(WEIGHTS_PATH, overwrite=True)
    print('save')


else:
    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)
