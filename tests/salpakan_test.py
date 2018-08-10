import unittest
from envs.salpakan_game import SalpakanGame, _parse_move
import numpy as np


class TestSalpakan(unittest.TestCase):

    def test_player_generator(self):
        game = SalpakanGame()
        # check if all troops are here
        self.assertEqual(0, np.sum(game.board))
        # check player 1
        self.assertEqual(137, np.sum(game.board[:, :4]))
        # check player 2
        self.assertEqual(-137, np.sum(game.board[:, 4:]))

    def test_parse_move(self):
        for i in range(289):
            move = _parse_move(i)
            print(move)


if __name__ == '__main__':
    unittest.main()