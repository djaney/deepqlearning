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
import envs

ENV_NAME = 'Salpakan-v0'
WEIGHTS_PATH = '.models/dqn_{}_weights.h5f'.format(ENV_NAME)
WINDOW_LENGTH = 4
MEMORY = 20000
WARM_UP = 1000



# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
input_shape = env.observation_space.shape
model = Sequential()
model.add(Conv2D(32, 4, activation='relu', input_shape=input_shape))
model.add(Conv2D(64, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='relu'))
model.summary()


train_mode = len(sys.argv) > 1 and sys.argv[1] == 'train'


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=MEMORY, window_length=WINDOW_LENGTH)
policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=WARM_UP,
               target_model_update=1e-2, policy=policy, enable_dueling_network=True,
               dueling_type='avg')
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
