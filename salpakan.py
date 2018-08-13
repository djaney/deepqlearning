#!/usr/bin/env python3
import numpy as np
import gym
import sys
import os
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, Reshape, multiply, Input
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
import envs


class SalpakanProcessor(Processor):
    def __init__(self, env):
        self.env = env

    def process_state_batch(self, batch):
        return [batch, np.stack([self.env.game.generate_mask()] * batch.shape[0])]


ENV_NAME = 'Salpakan-v0'
WEIGHTS_PATH = '.models/dqn_{}_weights.h5f'.format(ENV_NAME)
WINDOW_LENGTH = 1
NB_STEPS = 50000
MEMORY = 20000
WARM_UP = 100

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
input_shape = (WINDOW_LENGTH,) + env.observation_space.shape
input_layer = Input(shape=input_shape)
mask = Input(shape=(nb_actions,))

rehsape_layer = Reshape((9, 8, 3))(input_layer)
conv_1 = Conv2D(32, 4, activation='relu')(rehsape_layer)
conv_2 = Conv2D(192, 2, activation='relu')(conv_1)
flat_layer = Flatten()(conv_2)
dense_1 = Dense(512, activation='relu')(flat_layer)
output_layer = Dense(nb_actions)(dense_1)
masked_layer = multiply([output_layer, mask])

model = Model([input_layer, mask], masked_layer)
model.summary()

train_mode = len(sys.argv) > 1 and sys.argv[1] == 'train'

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=MEMORY, window_length=WINDOW_LENGTH)
policy = BoltzmannQPolicy()
processor = SalpakanProcessor(env)

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=WARM_UP,
               target_model_update=1e-2, policy=policy, processor=processor)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

if os.path.isfile(WEIGHTS_PATH) and os.access(WEIGHTS_PATH, os.R_OK):
    dqn.load_weights(WEIGHTS_PATH)

if train_mode:

    dqn.fit(env, nb_steps=NB_STEPS, visualize=True, verbose=2)
    dqn.save_weights(WEIGHTS_PATH, overwrite=True)
    print('save')


else:
    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)
