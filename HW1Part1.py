import numpy as np
import random
from turtle import Screen, Turtle
from numpy import savetxt, loadtxt
import time

class HeavyQueen:
    def __init__(self, chess_dim=8, file_name = None):
        self.n = chess_dim
        self.file_name = file_name
        self.chess_board = np.zeros((self.n, self.n)).astype(int)

    def init_borad(self):
        if self.file_name == None:
            for i in range(self.n):
                self.chess_board[random.randint(0, self.n-1), i] = random.randint(1, 9)
            savetxt('heavyqueen_init.csv', self.chess_board, delimiter=',')
        else:
            self.__loadboard()

    def __loadboard(self):
        load_mtx = loadtxt(self.file_name, delimiter=',').astype(int)
        self.n = len(load_mtx)
        self.chess_board = load_mtx

    def check_attack(self, row, col):
        count = 0
        for i in range(self.n):
            if self.chess_board[row,i]!=0:
                count = count + 1*(not(i==col))

        for i,j in zip(range(row-1,-1,-1),range(col-1,-1,-1)):
            if self.chess_board[i,j]!=0:
                count = count + 1

        for i,j in zip(range(row+1,self.n,1),range(col-1,-1,-1)):
            if self.chess_board[i,j]!=0:
                count = count + 1

        for i,j in zip(range(row-1,-1,-1),range(col+1,self.n,1)):
            if self.chess_board[i,j]!=0:
                count = count + 1

        for i,j in zip(range(row+1,self.n,1),range(col+1,self.n,1)):
            if self.chess_board[i,j]!=0:
                count = count + 1
        return count

    def sort_queen(self):
        np_index_list = np.transpose(np.nonzero(self.chess_board))
        weight_list = []
        pos_list = []
        for i_index in range(np_index_list.shape[0]):
            weight_list.append(self.chess_board[np_index_list[i_index,0],np_index_list[i_index,1]])
            pos_list.append(np_index_list[i_index,:])

        temp = sorted(zip(weight_list, pos_list), key=lambda x:x[0])
        queen_weight, queen_pos = map(list, zip(*temp))

        return queen_weight, queen_pos




    def test(self):
        print(self.n)
        pass


class DrawBoard:
########################################################################################
# class for drawing board, you can play with it. Good for visualization.
########################################################################################
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
                    text_value = 'Q' + str(num_value)
                    self.marker.write(text_value, align='center', font=self.FONT)

            self.greg.goto(-self.size * side/2, self.size*side/2 - self.size*(i+1))

    def drawchessboard(self):
        if self.is_plot:
            screen = Screen()
            # print(screen.screensize())
            screen.screensize(canvwidth=1000, canvheight=1000)
            screen.setup(1000,1000)
            screen.title('Heavy Queen')
            # screen.bgcolor('blue')
            screen.tracer(False)
            self.__chessboard()
            screen.mainloop()
        else:
            print('No plot! Set is_plot to True')

if __name__ == "__main__":
    start_time = time.time()
    heavy_queen = HeavyQueen(chess_dim=8)
    heavy_queen.init_borad()
    print("Runtime:  %s seconds" % (time.time() - start_time))

    mtx = heavy_queen.chess_board

    a, b = heavy_queen.sort_queen()

    ### draw chess board, please put it at the end of main
    draw_board = DrawBoard(value_list=heavy_queen.chess_board, is_plot=True,size=40)
    draw_board.drawchessboard()
