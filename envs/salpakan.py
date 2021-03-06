from gym import Env, spaces
import numpy as np
from .salpakan_game import SalpakanGame, Renderer, \
    MOVE_NORMAL, MOVE_CAPTURE, MOVE_CAPTURE_LOSE, MOVE_WIN, MOVE_PASS, MOVE_INVALID

OBSERVATION_SHAPE = (9, 8, 3)
MAX_STEPS = 200


class SalpakanEnv(Env):

    def __init__(self):

        self.observation_space = spaces.Box(low=0, high=1, shape=OBSERVATION_SHAPE, dtype=np.float16)
        self.action_space = spaces.Discrete(8 * 9 * 4 + 1)
        self.game = None
        self.view = None
        self.canvas = None
        self.steps = 0
        self.renderer = Renderer()
        self.done = False

    def step(self, action):
        move_type = self.game.move(action)
        ob = self._get_state()
        done = self.game.winner is not None or self.steps > MAX_STEPS
        self.done = self.done or done
        self.steps += 1

        if move_type == MOVE_NORMAL:
            reward = 1
        elif move_type == MOVE_CAPTURE:
            reward = 3
        elif move_type == MOVE_CAPTURE_LOSE:
            reward = -3
        elif move_type == MOVE_WIN:
            reward = 10
        elif move_type == MOVE_PASS:
            reward = 0
        elif move_type == MOVE_INVALID:
            reward = -10
            self.done = self.done or True
        else:
            reward = 0

        return ob, reward, self.done, {}

    def reset(self):
        self.done = False
        self.steps = 0
        self.game = SalpakanGame()
        return self._get_state()

    def render(self, mode='human'):
        self.renderer.render(self.game)

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
        observation[:, :, 0] = (np.clip(enemy_troops, 0, 1) * board[:, :, 1]) / 16
        # my units, clip troops channel
        observation[:, :, 1] = my_troops / 16
        # my spies
        observation[:, :, 2] = np.clip(my_troops, 0, 1) * board[:, :, 2]

        return observation
