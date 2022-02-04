import numpy as np

class HeavyQueen9N8:
    def __init__(self, chess_dim=8):
        self.chess_dim = chess_dim

    def test(self):
        print(self.chess_dim)
        pass

if __name__ == "__main__":
    class_a = HeavyQueen9N8(chess_dim=9)
    class_a.test()
    print('test')