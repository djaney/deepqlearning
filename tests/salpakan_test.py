import unittest
from envs.salpakan_game import SalpakanGame, _parse_move, _normalize_board, TROOP_FIVE_STAR, MOVE_CAPTURE, MOVE_CAPTURE_LOSE
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

    def test_normalize(self):
        game = SalpakanGame()
        board = np.copy(game.board)
        anti_board = np.copy(_normalize_board(-1, game.board))
        clash_board = board[:, :, 0] + anti_board[:, :, 0]
        min = np.min(clash_board)
        max = np.max(clash_board)
        self.assertEqual(0, min)
        self.assertEqual(0, max)


    def test_capture(self):
        game = SalpakanGame()
        game.board[:, :, 0] = 0
        game.board[0, 0, 0] = (TROOP_FIVE_STAR) * -1
        game.board[1, 0, 0] = (TROOP_FIVE_STAR - 1)
        game.turn = 1
        capture = game.move(2)
        self.assertEqual(MOVE_CAPTURE, capture)

    def test_capture_lose(self):
        game = SalpakanGame()
        game.board[:, :, 0] = 0
        game.board[0, 0, 0] = (TROOP_FIVE_STAR - 1) * -1
        game.board[1, 0, 0] = TROOP_FIVE_STAR
        game.turn = 1
        capture = game.move(2)
        self.assertEqual(MOVE_CAPTURE_LOSE, capture)

    def test_mask_generator(self):
        game = SalpakanGame()
        game.board[:, :, 0] = 0
        game.board[0, 0, 0] = (TROOP_FIVE_STAR - 1) * -1
        game.board[1, 0, 0] = TROOP_FIVE_STAR
        game.turn = 1
        mask = game.generate_mask()
        self.assertEqual(1, mask[2])
        self.assertEqual(1, mask[4])

if __name__ == '__main__':
    unittest.main()
