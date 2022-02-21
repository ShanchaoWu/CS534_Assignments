# // =================================================
# /*
# \author Haoying Zhou
# */
# // =================================================

import numpy as np
import random
from queue import PriorityQueue
from turtle import Screen, Turtle
from numpy import savetxt, loadtxt
import time

# find size of queue: a.qsize()
# add to queue: a.put()
# remove and get from queuee: a.get()

class HeavyQueen:
    def __init__(self, chess_dim=8, file_name = None):
        self.n = chess_dim
        self.file_name = file_name
        self.chess_board = np.zeros((self.n, self.n)).astype(int)
        self.lightest_weight = 0
        self.cost = 0
        self.h = 0
        self.g = 0
        self.node = 0
        self.branch = 0
        self.chess_board_old = []

    #### Board initialization
    def init_borad(self):
        if self.file_name == None:
            for i in range(self.n):
                self.chess_board[random.randint(0, self.n-1), i] = random.randint(1, 9)
            savetxt('heavyqueen_init.csv', self.chess_board, delimiter=',')
        else:
            self.__loadboard()
        self.chess_board_old = self.chess_board

    #### Load board from csv file
    def __loadboard(self):
        load_mtx = loadtxt(self.file_name, delimiter=',').astype(int)
        self.n = len(load_mtx)
        self.chess_board = load_mtx


    #### check how many queens attack one specific queen
    #### return the number of queens which attack the specific queen
    def check_attack(self, chess_board,row, col):
        count = 0
        for i in range(self.n):
            if chess_board[row,i]!=0:
                count = count + 1*(not(i==col))

        for i,j in zip(range(row-1,-1,-1),range(col-1,-1,-1)):
            if chess_board[i,j]!=0:
                count = count + 1

        for i,j in zip(range(row+1,self.n,1),range(col-1,-1,-1)):
            if chess_board[i,j]!=0:
                count = count + 1

        for i,j in zip(range(row-1,-1,-1),range(col+1,self.n,1)):
            if chess_board[i,j]!=0:
                count = count + 1

        for i,j in zip(range(row+1,self.n,1),range(col+1,self.n,1)):
            if chess_board[i,j]!=0:
                count = count + 1
        return count

    #### find all positions and weights for the queens and sort it with the weight
    #### return two list: queen_weight and queen_pos
    #### queen_weight[i] = weight of the queen (int)
    #### queen_pos[i] = position of the queen in chess board, 2X1 numpy array
    def sort_queen(self):
        np_index_list = np.transpose(np.nonzero(self.chess_board))
        weight_list = []
        pos_list = []
        for i_index in range(np_index_list.shape[0]):
            weight_list.append(self.chess_board[np_index_list[i_index,0],np_index_list[i_index,1]])
            pos_list.append(np_index_list[i_index,:])

        temp = sorted(zip(weight_list, pos_list), key=lambda x:x[0])
        queen_weight, queen_pos = map(list, zip(*temp))
        self.lightest_weight = queen_weight[0]
        return queen_weight, queen_pos

    def find_weight(self, board):
        np_index_list = np.transpose(np.nonzero(board))
        weight_list = []
        pos_list = []
        for i_index in range(np_index_list.shape[0]):
            weight_list.append(board[np_index_list[i_index, 0], np_index_list[i_index, 1]])
            pos_list.append(np_index_list[i_index, :])
        return weight_list, pos_list

    #### check how many pair of queens attacking each other
    #### return the number of pairs
    def check_total(self, chess_board):
        queen_weight, queen_pos = self.sort_queen()
        count = 0
        for i_queen in range(len(queen_pos)):
            row = queen_pos[i_queen][0]
            col = queen_pos[i_queen][1]
            queen_attack = self.check_attack(chess_board, row, col)
            count = count + queen_attack
        return int(count/2)

    def cal_heuristic(self, attack_pair):
        return self.lightest_weight ^ 2 * attack_pair

    def cal_g(self, board):
        cost = 0

        weight_list_new, pos_list_new = self.find_weight(board)
        weight_list_old, pos_list_old = self.find_weight(board)

        for i in range(len(weight_list_new)):
            assert weight_list_old[i] != weight_list_new[i] , 'wrong algorithm'

            weight = weight_list_new[i]
            pos_new = pos_list_new[i]
            pos_old = pos_list_old[i]

            step = abs(pos_new[0] - pos_old[0])

            cost = cost + weight*weight*step

        return cost


    def greedy_search_test(self):
        while(True):
            queen_weight, queen_pos = self.sort_queen()
            attack_pair = self.check_total(self.chess_board)
            for i_index in range(len(queen_weight)):
                weight = queen_weight[i_index]
                row = queen_pos[i_index][0]
                col = queen_pos[i_index][1]
                row_new = row
                self.node = self.node + 1
                for i_pos in range(self.n):
                    self.branch = self.branch + 1
                    if self.check_total(self.chess_board) == 0:
                        print('done')
                        return
                    if i_pos != row:
                        self.chess_board[row, col] = 0
                        self.chess_board[i_pos, col] = weight
                        current_pair = self.check_total(self.chess_board)
                        if current_pair > attack_pair:
                            self.chess_board[row, col] = weight
                            self.chess_board[i_pos, col] = 0
                        else:
                            attack_pair = current_pair
                            row_new = i_pos
                    if row_new != row:
                        self.chess_board[row, col] = 0
                self.cost = self.cost + self.cal_heuristic(attack_pair)
                # self.cost = self.cost + self.cal_g(self.chess_board)# abs(row_new - row)

    def A_star(self):
        pass



    def run(self):
        self.greedy_search_test()
        # self.A_star()


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
            screen.setup(1000, 1000)
            screen.screensize(canvwidth=1000, canvheight=1000)
            screen.title('Heavy Queen')
            # screen.bgcolor('blue')
            screen.tracer(False)
            self.__chessboard()
            screen.mainloop()
        else:
            print('No plot! Set is_plot to True')

if __name__ == "__main__":
    start_time = time.time()

    # heavy_queen = HeavyQueen(chess_dim=8, file_name='heavyqueen_init.csv')
    # print('Enter the dimension of the chess board: \n')
    # N = input()
    # N = int(N)
    N = 16
    heavy_queen = HeavyQueen(chess_dim=N, file_name='heavyqueen9n8_init.csv')

    cost = []
    node = []
    branch = []
    b_factor = []
    time_run = []

    # for i in range(10):
    #     heavy_queen.init_borad()
    #     start_time = time.time()
    #     heavy_queen.run()
    #     time_run.append(time.time() - start_time)
    #     node.append(heavy_queen.node)
    #     branch.append(heavy_queen.branch)
    #     b_factor.append(heavy_queen.branch/heavy_queen.node)


    heavy_queen.init_borad()
    heavy_queen.run()
    print("Runtime:  %s seconds" % (time.time() - start_time))
    # mtx = heavy_queen.chess_board
    print(heavy_queen.cost)
    print(heavy_queen.node)
    print(heavy_queen.branch)
    print(heavy_queen.branch/heavy_queen.node)

    ## draw chess board, please put it at the end of main
    plot_cub_size = int(800/N)
    draw_board = DrawBoard(value_list=heavy_queen.chess_board, is_plot=True, size=plot_cub_size)
    draw_board.drawchessboard()
