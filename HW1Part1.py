import numpy as np
import random

class HeavyQueen:
    def __init__(self, chess_dim=8):
        self.n = chess_dim
        self.chess_board = np.zeros((self.n, self.n))

    def init_borad(self):
        for i in range(self.n):
            self.chess_board[random.randint(0, self.n-1), i] = random.randint(1, 9)

    def test(self):
        print(self.n)
        pass

if __name__ == "__main__":
    heave_queen = HeavyQueen(chess_dim=8)
    heave_queen.init_borad()
    print(heave_queen.chess_board)