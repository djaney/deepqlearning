import numpy as np
import random


PIECE_CONF = [1, 2, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
WIDTH = 9
HEIGHT = 8
class SalpakanGame:
    def __init__(self):
        # Create board
        self.board = np.zeros((9, 8))

        # generate troops
        self._generate_troops(0)
        self._generate_troops(1)

        self.turn = random.randint(0, 1)

    def _generate_troops(self, player):
        assert player == 0 or player == 1

        if player == 0:
            y_min = 0
            y_max = HEIGHT / 2 -1
        else:
            y_min = HEIGHT / 2
            y_max = HEIGHT - 1

        for (i, p) in enumerate(PIECE_CONF):
            for r in range(p):
                for _ in range(36):
                    y = random.randint(y_min, y_max)
                    x = random.randint(0, WIDTH-1)
                    if self.board[x][y] == 0:
                        self.board[x][y] = i+1 if player == 0 else (i+1)*-1
                        break

    def play(self, player, move):
        assert self.turn == player
