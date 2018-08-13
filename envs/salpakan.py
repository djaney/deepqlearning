from gym import Env, spaces
import numpy as np
from .salpakan_game import SalpakanGame
import tkinter as tk
import time

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

    def step(self, action):
        reward = self.game.move(action)
        ob = self._get_state()
        done = self.game.winner is not None or self.steps >= MAX_STEPS
        self.steps += 1
        return ob, reward, done, {}

    def reset(self):
        self.game = SalpakanGame()
        return self._get_state()

    def render(self, mode='human'):

        width = 562
        height = 500

        x_tiles = 9
        y_tiles = 8

        tile_width = width / x_tiles
        tile_height = height / y_tiles

        if self.view is None:
            self.view = tk.Tk()
            self.view.geometry('{}x{}'.format(width, height))
            self.view.resizable(width=False, height=False)
            self.canvas = tk.Canvas(self.view, width=width, height=height)
            self.canvas.pack()

        # clear
        self.canvas.delete("all")

        # set board to white
        self.canvas.create_rectangle(0, 0, width, height, fill='white')

        # add lines
        for i in range(x_tiles):
            self.canvas.create_line(tile_width * i, 0, tile_width * i, height)
        for i in range(y_tiles):
            self.canvas.create_line(0, tile_height * i, width, tile_height * i)

        # Draw cells
        for x, col in enumerate(self.game.board):
            for y, cell in enumerate(col):
                x1 = tile_width * x
                y1 = tile_height * y
                x2 = tile_width * x + tile_width
                y2 = tile_height * y + tile_height
                # Draw pieces
                if cell[0] != 0:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='red' if cell[0] > 0 else 'black')
                    self.canvas.create_text(x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2,
                                            fill='white',
                                            font="Times 20 bold",
                                            text=str(cell[0]))
        self.view.update_idletasks()
        # time.sleep(1)

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
