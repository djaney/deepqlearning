import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np


class Agent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.done = False
        self.episodes = 5000
        self.sample_size = 32
        self.session = deque(maxlen=2000)

        self.epsilon = 0.7
        self.gamma = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._create_model()

    def _create_model(self):
        # create model
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_size, ), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def act(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = np.argmax(self.model.predict(state)[0])
        return action

    def remember(self, state, action, reward, next_state, done):
        self.session.append((state, action, reward, next_state, done))

    def train(self, sample_size):
        if len(self.session) <= sample_size:
            return False

        samples = random.sample(self.session, sample_size)

        for state, action, reward, state_, done in samples:

            if done:
                q = reward
            else:
                q = (reward + self.gamma * np.max(self.model.predict(state_)[0]))

            # get current q value
            target = self.model.predict(state)
            # update q value
            target[0][action] = q
            # save new q value
            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return True