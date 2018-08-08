import numpy as np
import random
import math

PIECE_CONF = [1, 2, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
WIDTH = 9
HEIGHT = 8
SPECIAL_MOVES_N = 1

MOVE_NORMAL = 0
MOVE_CAPTURE = 1
MOVE_WIN = 2

CHANNEL_TROOPS = 0
CHANNEL_PERCEPTION = 1

PLAYER_1 = 0
PLAYER_2 = 1


def _parse_move(move):
    direction = (move - SPECIAL_MOVES_N) % 4
    square_id = math.floor((move - SPECIAL_MOVES_N) / 4)
    x = square_id % HEIGHT
    y = math.floor(square_id / WIDTH)

    if direction == 0:
        _x = x - 1
        _y = y
    elif direction == 1:
        _x = x + 1
        _y = y
    elif direction == 2:
        _x = x
        _y = y - 1
    elif direction == 3:
        _x = x
        _y = y + 1
    else:
        raise Exception("Invalid direction")
    return square_id, x, y, _x, _y, direction


def _normalize_board(player, board):
    if player < 0:
        return board * -1
    else:
        return board * 1


class SalpakanGame:
    def __init__(self):
        # Create board
        # width x height x channels
        # channels = troops, perception
        self.board = np.zeros((9, 8, 2))

        # generate troops
        self._generate_troops(0)
        self._generate_troops(1)

        self.turn = random.randint(0, 1)

    def _generate_troops(self, player):
        assert player == PLAYER_1 or player == PLAYER_2

        if player == 0:
            y_min = 0
            y_max = HEIGHT / 2 - 1
        else:
            y_min = HEIGHT / 2
            y_max = HEIGHT - 1

        for (i, p) in enumerate(PIECE_CONF):
            for r in range(p):
                for _ in range(36):
                    y = random.randint(y_min, y_max)
                    x = random.randint(0, WIDTH-1)
                    # if no troop
                    if self.board[x][y][CHANNEL_TROOPS] == 0:
                        # set troops, negative if player 2
                        self.board[x][y][CHANNEL_TROOPS] = i+1 if player == PLAYER_1 else (i+1)*-1
                        break

    def move(self, player, move):
        """
        direction: 0=left, 1=right, 2=top, 3=bottom
        :param player:
        :param move:
        :return:
        """
        assert self.turn == player

        if move != 0:
            square_id, x, y, _x, _y, direction = _parse_move(move)

            assert self._is_valid_move(player, (x, y), (_x, _y))

            # TODO
            # do move

        # change turn
        self.turn = PLAYER_2 if self.turn == PLAYER_1 else PLAYER_1

        # return move type

    def _is_valid_move(self, player, src, destination):
        board = _normalize_board(player, self.board)

        # check if moving own piece
        if board[src[0]][src[1]][CHANNEL_TROOPS] <= 0:
            return False

        # check if destination not capturing own
        if board[destination[0]][destination[1]][CHANNEL_TROOPS] > 0:
            return False

        return True

    def _get_move_type(self, player, src, destination):
        board = _normalize_board(player, self.board)
        me = board[src[0]][src[1]][CHANNEL_TROOPS]
        him = board[destination[0]][destination[1]][CHANNEL_TROOPS]