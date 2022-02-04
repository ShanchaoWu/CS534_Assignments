import numpy as np
import random
from turtle import Screen, Turtle

class HeavyQueen:
    def __init__(self, chess_dim=8):
        self.n = chess_dim
        self.chess_board = np.zeros((self.n, self.n)).astype(int)

    def init_borad(self):
        for i in range(self.n):
            self.chess_board[random.randint(0, self.n-1), i] = random.randint(1, 9)

    def test(self):
        print(self.n)
        pass


class DrawBoard:
    def __init__(self, value_list = np.zeros((8, 8)).astype(int), is_plot = False, size = 100):
        self.value_list = value_list
        self.is_plot = is_plot
        self.n = len(self.value_list)
        self.FONT_SIZE = 18
        self.FONT = ('Arial', self.FONT_SIZE, 'normal')
        self.size = size
        self.greg = Turtle()
        self.greg.hideturtle()

        self.marker = Turtle()
        self.marker.penup()
        self.marker.hideturtle()

    def __square(self, color):
        self.greg.fillcolor(color)
        self.greg.pendown()
        self.greg.begin_fill()

        for _ in range(4):
            self.greg.forward(self.size)
            self.greg.left(90)

        self.greg.end_fill()

        self.greg.penup()
        self.greg.forward(self.size)

    def __chessboard(self):
        side = len(self.value_list)

        self.greg.penup()
        self.greg.goto(-self.size * side/2, self.size * side/2)

        for i in range(side):
            for j in range(side):
                num_value = self.value_list[i,j]
                if num_value == 0:
                    text, color = ('black', 'white')
                else:
                    text, color = ('white', 'black')
                self.__square(color)
                self.marker.goto(self.greg.xcor() - self.size/2, self.greg.ycor() + self.size/2 - self.FONT_SIZE/2)
                self.marker.pencolor(text)

                if num_value != 0:
                    text_value = 'Q-' + str(num_value)
                    self.marker.write(text_value, align='center', font=self.FONT)

            self.greg.goto(-self.size * side/2, self.size*side/2 - self.size*(i+1))

    def drawchessboard(self):
        if self.is_plot:
            screen = Screen()
            # screen.screensize(canvwidth=800,canvheight=600)
            screen.title('Heavy Queen')
            # screen.bgcolor('blue')
            screen.tracer(False)
            self.__chessboard()
            screen.mainloop()
        else:
            print('No plot! Set is_plot to True')

if __name__ == "__main__":
    heave_queen = HeavyQueen(chess_dim=8)
    heave_queen.init_borad()
    draw_board = DrawBoard(value_list=heave_queen.chess_board, is_plot=True)
    draw_board.drawchessboard()