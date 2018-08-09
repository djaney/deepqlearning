from gym import Env, spaces
import numpy as np
from .salpakan_game import SalpakanGame

OBSERVATION_SHAPE = (9, 8, 3)

class SalpakanEnv(Env):

    def __init__(self):

        self.observation_space = spaces.Box(low=0, high=1, shape=OBSERVATION_SHAPE, dtype=np.float16)
        self.action_space = spaces.Discrete(8*9*4+1)
        self.game = None

    def step(self, action):
        reward = self.game.move(action)
        ob = self._get_state()
        done = self.game.winner is not None
        return ob, reward, done, {}

    def reset(self):
        self.game = SalpakanGame()
        return self._get_state()

    def render(self, mode='human'):
        print('hehe')

    def close(self):
        super().close()

    def seed(self, seed=None):
        return super().seed(seed)

    def _get_state(self):

        observation = np.zeros(shape=OBSERVATION_SHAPE)

        board = self.game.get_board()

        my_troops = np.clip(board[:, :, 0], 0, None)
        enemy_troops = np.clip(board[:, :, 0] * -1, 0, None)

        # enemy perception, flip and clip, troops channel
        observation[:, :, 0] = np.clip(enemy_troops, 0, 1) * board[:, :, 1]
        # my units, clip troops channel
        observation[:, :, 1] = my_troops
        # my spies
        observation[:, :, 2] = np.clip(my_troops, 0, 1) * board[:, :, 2]

        return observation
