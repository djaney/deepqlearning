import random
from collections import deque
import numpy as np
import os
from keras.models import load_model


class Agent:

    def __init__(self,
                 state_size,
                 action_size,
                 memory_size=2000,
                 epsilon=1,
                 gamma=0.9,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 learning_rate=0.001,
                 model_path=None):

        self.state_size = state_size
        self.action_size = action_size
        self.session = deque(maxlen=memory_size)

        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._create_model()
        self.reward_running_avg = deque(maxlen=memory_size)
        self.training_sessions = 0

        if model_path is not None and os.path.isfile(model_path) and os.access(model_path, os.R_OK):
            self.model.load_weights(model_path)

    def _create_model(self):
        raise NotImplemented()

    def act(self, state):
        state = self.format_state(state)
        if random.uniform(0, 1) <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = np.argmax(self.model.predict(np.expand_dims(state, 0))[0])
        return action

    def remember(self, state, action, reward, next_state, done):
        state = self.format_state(state)
        next_state = self.format_state(next_state)
        self.reward_running_avg.append(reward)
        self.session.append((state, action, reward, next_state, done))

    def format_state(self, state):
        raise NotImplemented()

    def train(self, sample_size, verbose=0):
        if len(self.session) <= sample_size:
            return False

        samples = random.sample(self.session, sample_size)
        x = []
        y = []
        for state, action, reward, state_, done in samples:

            if done:
                q = reward
            else:
                q = (reward + self.gamma * np.max(self.model.predict(np.expand_dims(state_, 0))[0]))

            # get current q value
            target = self.model.predict(np.expand_dims(state_, 0))[0]
            # update q value
            target[action] = q
            # save new q value
            x.append(state)
            y.append(target)
        self.model.fit(np.array(x), np.array(y), epochs=1, verbose=verbose)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.training_sessions += 1

        return True

    def save(self, name):
        self.model.save_weights(name)

    def get_average_reward(self):
        return np.average(self.reward_running_avg)
