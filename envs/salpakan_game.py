import numpy as np
import random
import math

PIECE_CONF = [1, 2, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
WIDTH = 9
HEIGHT = 8
SPECIAL_MOVES_N = 1

MOVE_NORMAL = 0
MOVE_CAPTURE = 1
MOVE_WIN = 10
MOVE_PASS = 0

CHANNEL_TROOPS = 0
CHANNEL_PERCEPTION = 1
CHANNEL_SPY_PERCEPTION = 2

PLAYER_1 = 0
PLAYER_2 = 1

TROOP_FLAG = 1
TROOP_SPY = 2
TROOP_PRIVATE = 3
TROOP_FIVE_STAR = 15


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
    n_board = np.copy(board)
    if player < 0:
        n_board[:, :, CHANNEL_TROOPS] = n_board[:, :, CHANNEL_TROOPS] * -1
    return n_board


class SalpakanGame:
    def __init__(self):
        # Create board
        # width x height x channels
        # channels = troops, perception
        self.board = np.zeros((9, 8, 3))

        # generate troops
        self._generate_troops(0)
        self._generate_troops(1)

        self.turn = random.randint(0, 1)

        self.winner = None

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
                    x = random.randint(0, WIDTH - 1)
                    # if no troop
                    if self.board[x][y][CHANNEL_TROOPS] == 0:
                        # set troops, negative if player 2
                        self.board[x][y][CHANNEL_TROOPS] = i + 1 if player == PLAYER_1 else (i + 1) * -1
                        # set spy flag
                        if self.board[x][y][CHANNEL_TROOPS] == TROOP_SPY:
                            self.board[x][y][CHANNEL_SPY_PERCEPTION] = 1
                        break

    def move(self, move):
        """
        direction: 0=left, 1=right, 2=top, 3=bottom
        :param player:
        :param move:
        :return:
        """
        player = self.turn

        if move != 0:
            square_id, x, y, _x, _y, direction = _parse_move(move)

            assert self._is_valid_move(player, (x, y), (_x, _y))

            move_type = self._get_move_type(player, (x, y), (_x, _y))

            if move_type == MOVE_NORMAL:
                tmp = self.board[x][y]
                self.board[x][y] = self.board[_x][_y]
                self.board[_x][_y] = tmp
            elif move_type == MOVE_CAPTURE:
                normalized_board = _normalize_board(player, self.board)
                me = normalized_board[x][y][CHANNEL_TROOPS]
                him = normalized_board[_x][_y][CHANNEL_TROOPS]

                if me == him:  # cancel out
                    win = 0
                elif me == TROOP_SPY and him != -TROOP_PRIVATE:  # spy captures
                    win = 1
                elif me == TROOP_PRIVATE and him == -TROOP_SPY:  # private captures spy
                    win = 1
                else:  # normal rank based clash
                    win = 1 if me > him else -1

                if win > 0:  # win
                    self.board[_x][_y] = self.board[x][y]
                    self.board[x][y] = self.board[x][y] * 0
                elif win < 0:  # lose
                    self.board[_x][_y][CHANNEL_PERCEPTION] = max(self.board[x][y][CHANNEL_TROOPS]+1,
                                                                 self.board[_x][_y][CHANNEL_PERCEPTION])
                    self.board[x][y] = self.board[x][y] * 0
                else:
                    self.board[_x][_y][CHANNEL_PERCEPTION] = self.board[x][y][CHANNEL_TROOPS]
                    self.board[x][y] = self.board[x][y] * 0
                    self.board[_x][_y] = self.board[_x][_y] * 0
            elif move_type == MOVE_WIN:
                self.board[_x][_y] = self.board[x][y]
                self.board[x][y] = self.board[x][y] * 0
                self.winner = player
        else:
            move_type = MOVE_PASS

        # change turn
        self.turn = PLAYER_2 if self.turn == PLAYER_1 else PLAYER_1

        # return move type
        return move_type

    def get_board(self):
        return _normalize_board(self.turn, self.board)

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

        if me > 0 and him == -TROOP_FLAG:
            return MOVE_WIN
        elif me > 0 and 0 > him:
            return MOVE_CAPTURE
        else:
            return MOVE_NORMAL

    def is_valid_move(self, move):
        square_id, x, y, _x, _y, direction = _parse_move(move)
        return self._is_valid_move(self.turn, (x, y), (_x, _y))