# // =================================================
# /*
# \author Shanchao Wu
# */
# // =================================================

import numpy as np
import random
from turtle import Screen, Turtle
from numpy import savetxt, loadtxt
import time
import math
import pandas as pd

class HeavyQueen9N8:
    def __init__(self, chess_dim=8, file_name='default.txt', load_file=None):
        self.dim = chess_dim
        self.file_name = file_name
        self.load_file = load_file
        self.chess_board = np.zeros((self.dim, self.dim)).astype(int)

    def init_board(self):
        if self.load_file == None:
            for i in range(self.dim):
                self.chess_board[random.randint(0, self.dim - 1), i] = random.randint(1, 9)
            flag = int(self.dim / 8)
            while (flag):
                col = random.randint(0, self.dim - 1)
                row = random.randint(0, self.dim - 1)
                while self.chess_board[row, col] != 0:
                    col = random.randint(0, self.dim - 1)
                    row = random.randint(0, self.dim - 1)
                self.chess_board[row, col] = random.randint(1, 9)
                flag -= 1
            savetxt(self.file_name, self.chess_board, delimiter=',')
            return self.chess_board
        else:
            self.load_board()
            return self.chess_board

    def load_board(self):
        load_mtx = loadtxt(self.load_file, delimiter=',').astype(int)
        self.dim = len(load_mtx)
        self.chess_board = load_mtx


class genetic_algo:
    def __init__(self, board, population=200, crossover=3, mutate=0.2, time=0, eli_ratio=0.2, cul_ratio=0.2,
                 max_gen=1000):
        self.generationNum = 0  # indicate which generation is now, initial = 0
        self.population = list()  # population in each generation
        self.pValue = population  # num of population in each generation, setted by user, default 100 population/generation
        self.cost_list = list()  # cost of each population
        self.crossover = crossover  # in which postion to crossover, default colume 3
        self.mutate = mutate  # the chance of mutation, default 10%
        self.initial_board = board  # initial board, need to input
        self.init_col_list = list(np.nonzero(board.T)[0])
        self.init_row_list = list(np.nonzero(board.T)[1])
        self.weight_list = self.init_weight_list()  # weight of each queen
        self.dim = len(board)  # get dim of board
        self.duration = time  # set how much time to run the algorithm
        self.queen_num = len(self.init_col_list)
        self.prob_list = list()  # probability of each population
        self.eli_num = int(self.pValue * eli_ratio)  # ratio of boards to be preserved in elitism, default 10
        self.cul_num = int(self.pValue * cul_ratio)  # ratio of boards to be deleted in culling, default 20
        self.next_population_list = list()  # collectting generated population
        self.max_generation = max_gen  # max num of generating, default 100
        self.max_time = time  # max seconds of time to run, default 0 means not set
        self.min_cost_list = list()  # a list of minimal cost in each generation
        self.min_pop_list = list()  # a list of minimal board in each generation
        self.flag = 0  # indicate the status of processing, if done flag = 1
        self.min_cost = 0
        self.min_pop = []
        self.process_time = 0
        self.time_list = []

    def attack_cost(self, row_list):  # calculate the cost of queens' attacks
        count = 0
        pos_row = row_list
        pos_col = self.init_col_list
        for i in range(len(pos_col)):
            row = pos_row[i]
            col = pos_col[i]
            # cal right
            for j in range(i + 1, len(pos_col)):
                if pos_row[j] == row:
                    count += 1
            # cal below
            for j in range(i + 1, len(pos_col)):
                if pos_col[j] == col:
                    count += 1
            # cal right-below
            for j in range(i + 1, len(pos_col)):
                col_next = pos_col[j]
                row_next = pos_row[j]
                diff_col = col_next - col
                if row + diff_col < self.dim:
                    if row_next == row + diff_col:
                        count += 1
                else:
                    break
            # cal right_above
            for j in range(i + 1, len(pos_col)):
                col_next = pos_col[j]
                row_next = pos_row[j]
                diff_col = col_next - col
                if row - diff_col > 0:
                    if row_next == row - diff_col:
                        count += 1
                else:
                    break
        return count * 100

    def init_weight_list(self):  # initial the weight list
        weight_list = []
        for i in range(len(self.init_col_list)):
            weight_list.append(self.initial_board[self.init_row_list[i]][self.init_col_list[i]])
        return weight_list

    def move_cost(self, row_list):  # calculate the move cost of current board with initial board
        cost = 0
        for i in range(len(self.init_col_list)):
            row = self.init_row_list[i]
            cur_row = row_list[i]
            if row != cur_row:
                cost = cost + math.pow(self.weight_list[i], 2) * abs(row - cur_row)
        return cost

    def rand_change_pos(self, i, row_list):  # change the (i+1)th queen position without changing input
        row_list_new = row_list
        row = row_list_new[i]
        rand_row = random.randint(0, self.dim - 1)
        while rand_row == row:
            rand_row = random.randint(0, self.dim - 1)
        row_list_new[i] = rand_row
        return row_list_new

    def init_environment(self):  # randomly change 1-2 queens in their column, and get the initial populations
        self.population.append(self.init_row_list)
        i = 1
        while i < self.pValue:
            # randomly select 1-2 colume to change queens' position
            num_q = random.randint(1, 2)
            if num_q == 1:
                queen = random.randint(0, self.queen_num - 1)
                row_list_new = self.rand_change_pos(queen, self.init_row_list[:])
            else:
                queen1 = random.randint(0, self.queen_num - 1)
                queen2 = random.randint(0, self.queen_num - 1)
                while queen1 == queen2:
                    queen2 = random.randint(0, self.queen_num - 1)
                row_list_new = self.rand_change_pos(queen1, self.init_row_list[:])
                row_list_new = self.rand_change_pos(queen2, row_list_new)
            if row_list_new not in self.population:
                i += 1
                self.population.append(row_list_new)

    def cal_cost_list(self):  # calculate the cost list using row list in populations
        if 0 != len(self.cost_list):
            self.cost_list = list()
        for i in self.population:
            cost = 0
            move_cost = self.move_cost(i)
            attack_cost = self.attack_cost(i)
            cost = move_cost + attack_cost
            self.cost_list.append(int(cost))

    def cal_prob(self,
                 cost_list):  # calculate the random selecting probability of each row list in population using its cost
        prob_list = list()
        temp = pd.DataFrame({'cost': cost_list})
        invert_cost_list = list(sum(temp['cost']) / temp['cost'])
        total_invert_cost = sum(invert_cost_list)
        for i in invert_cost_list:
            prob = i / total_invert_cost
            prob_list.append(prob)
        return prob_list

    def sort_cost_pop_list(self):  # sort population list according cost list in ascending order
        data = pd.DataFrame({"c": self.cost_list, "p": self.population})
        data_sorted = data.sort_values(by=['c'])
        cost_sorted = list(data_sorted['c'])
        population_sorted = list(data_sorted['p'])
        return (cost_sorted, population_sorted)

    def elitism(self):  # preserve best eli_num populations
        cost_sorted, population_sorted = self.sort_cost_pop_list()
        self.next_population_list.extend(population_sorted[:self.eli_num])

    def culling(self):  # delete worst cul_num populations
        cost_sorted, population_sorted = self.sort_cost_pop_list()
        cul_population = population_sorted[:len(population_sorted) - self.cul_num]
        cul_cost = cost_sorted[:len(population_sorted) - self.cul_num]
        return (cul_cost, cul_population)

    def get_parent(self, prob_list, population_list):  # randomly choose a population as parent
        prob = random.random()
        count = 0
        par = None
        for i in range(len(prob_list)):
            count += prob_list[i]
            if count >= prob:
                par = population_list[i]
                return par

    def cross_over(self, par1, par2):  # give two populations to generate two new populations
        child1 = list()
        child2 = list()
        child1 = par1[:self.crossover] + par2[self.crossover:]
        child2 = par2[:self.crossover] + par1[self.crossover:]
        return (child1, child2)

    def mutation(self, child):  # randomly mutation in child
        if self.mutate > 0:
            pro = random.random()
            if pro <= self.mutate:
                pos = random.randint(0, self.queen_num - 1)
                child = self.rand_change_pos(pos, child)

    def processing(self):  # mean part of genetic algorithm working
        if self.flag == 0:  # first time to process
            start_time = time.time()
            if 0 == len(self.population):
                # print("=====================Initializing===========================")
                self.init_weight_list()  # initial weight list
                self.init_environment()  # initial populations
                self.cal_cost_list()  # generate cost list for elistism and culling
                self.time_list.append(0)
            while self.generationNum < self.max_generation:
                self.elitism()  # elitism
                cul_cost, cul_population = self.culling()  # get cost and populations after culling
                self.min_cost_list.append(cul_cost[0])
                self.min_pop_list.append(cul_population[0])
                prob_list = self.cal_prob(cul_cost)  # calculate probability of current populations
                while len(
                        self.next_population_list) < self.pValue:  # while population not reach setted num, generate child and put it in list
                    par1 = self.get_parent(prob_list, cul_population)
                    par2 = self.get_parent(prob_list, cul_population)
                    child1, child2 = self.cross_over(par1, par2)
                    self.mutation(child1)
                    self.mutation(child2)
                    self.next_population_list.append(child1)
                    self.next_population_list.append(child2)
                self.population = self.next_population_list  # update population into next generation
                self.next_population_list = list()  # clean next population list
                self.cal_cost_list()
                self.generationNum += 1
                end_time = time.time()
                self.time_list.append(end_time - start_time)
                if self.max_time != 0:  # checking if time of running setted
                    if end_time - start_time >= self.max_time:  # check if have reach setted running time
                        break

            cost, pop = self.sort_cost_pop_list()
            self.min_cost_list.append(cost[0])
            self.min_pop_list.append(pop[0])
            self.min_cost = self.min_cost_list[-1]
            self.min_pop = self.min_pop_list[-1]
            self.process_time = end_time - start_time
            self.flag = 1
            # print('=====================Finishing===========================')
        else:  # for re-processing
            self.clean()
            self.processing()

    def clean(self):  # when finish, set augment to initial state, for user to run ag again.
        self.generationNum = 0
        self.population = list()
        self.cost_list = list()
        self.prob_list = list()
        self.min_cost_list = list()
        self.min_pop_list = list()
        self.flag = 0

    def get_board(self, row_list):  # form a numpy-array
        board = np.zeros((self.dim, self.dim)).astype(int)
        for i in range(len(self.init_col_list)):
            row = row_list[i]
            col = self.init_col_list[i]
            board[row][col] = self.weight_list[i]
        return board


class genetic_algo:
    def __init__(self, board, population=200, crossover=3, mutate=0.2, time=0, eli_ratio=0.2, cul_ratio=0.2,
                 max_gen=1000):
        self.generationNum = 0  # indicate which generation is now, initial = 0
        self.population = list()  # population in each generation
        self.pValue = population  # num of population in each generation, setted by user, default 100 population/generation
        self.cost_list = list()  # cost of each population
        self.crossover = crossover  # in which postion to crossover, default colume 3
        self.mutate = mutate  # the chance of mutation, default 10%
        self.initial_board = board  # initial board, need to input
        self.init_col_list = list(np.nonzero(board.T)[0])
        self.init_row_list = list(np.nonzero(board.T)[1])
        self.weight_list = self.init_weight_list()  # weight of each queen
        self.dim = len(board)  # get dim of board
        self.duration = time  # set how much time to run the algorithm
        self.queen_num = len(self.init_col_list)
        self.prob_list = list()  # probability of each population
        self.eli_num = int(self.pValue * eli_ratio)  # ratio of boards to be preserved in elitism, default 10
        self.cul_num = int(self.pValue * cul_ratio)  # ratio of boards to be deleted in culling, default 20
        self.next_population_list = list()  # collectting generated population
        self.max_generation = max_gen  # max num of generating, default 100
        self.max_time = time  # max seconds of time to run, default 0 means not set
        self.min_cost_list = list()  # a list of minimal cost in each generation
        self.min_pop_list = list()  # a list of minimal board in each generation
        self.flag = 0  # indicate the status of processing, if done flag = 1
        self.min_cost = 0
        self.min_pop = []
        self.process_time = 0
        self.time_list = []

    def attack_cost(self, row_list):  # calculate the cost of queens' attacks
        count = 0
        pos_row = row_list
        pos_col = self.init_col_list
        for i in range(len(pos_col)):
            row = pos_row[i]
            col = pos_col[i]
            # cal right
            for j in range(i + 1, len(pos_col)):
                if pos_row[j] == row:
                    count += 1
            # cal below
            for j in range(i + 1, len(pos_col)):
                if pos_col[j] == col:
                    count += 1
            # cal right-below
            for j in range(i + 1, len(pos_col)):
                col_next = pos_col[j]
                row_next = pos_row[j]
                diff_col = col_next - col
                if row + diff_col < self.dim:
                    if row_next == row + diff_col:
                        count += 1
                else:
                    break
            # cal right_above
            for j in range(i + 1, len(pos_col)):
                col_next = pos_col[j]
                row_next = pos_row[j]
                diff_col = col_next - col
                if row - diff_col > 0:
                    if row_next == row - diff_col:
                        count += 1
                else:
                    break
        return count * 100

    def init_weight_list(self):  # initial the weight list
        weight_list = []
        for i in range(len(self.init_col_list)):
            weight_list.append(self.initial_board[self.init_row_list[i]][self.init_col_list[i]])
        return weight_list

    def move_cost(self, row_list):  # calculate the move cost of current board with initial board
        cost = 0
        for i in range(len(self.init_col_list)):
            row = self.init_row_list[i]
            cur_row = row_list[i]
            if row != cur_row:
                cost = cost + math.pow(self.weight_list[i], 2) * abs(row - cur_row)
        return cost

    def rand_change_pos(self, i, row_list):  # change the (i+1)th queen position without changing input
        row_list_new = row_list
        row = row_list_new[i]
        rand_row = random.randint(0, self.dim - 1)
        while rand_row == row:
            rand_row = random.randint(0, self.dim - 1)
        row_list_new[i] = rand_row
        return row_list_new

    def init_environment(self):  # randomly change 1-2 queens in their column, and get the initial populations
        self.population.append(self.init_row_list)
        i = 1
        while i < self.pValue:
            # randomly select 1-2 colume to change queens' position
            num_q = random.randint(1, 2)
            if num_q == 1:
                queen = random.randint(0, self.queen_num - 1)
                row_list_new = self.rand_change_pos(queen, self.init_row_list[:])
            else:
                queen1 = random.randint(0, self.queen_num - 1)
                queen2 = random.randint(0, self.queen_num - 1)
                while queen1 == queen2:
                    queen2 = random.randint(0, self.queen_num - 1)
                row_list_new = self.rand_change_pos(queen1, self.init_row_list[:])
                row_list_new = self.rand_change_pos(queen2, row_list_new)
            if row_list_new not in self.population:
                i += 1
                self.population.append(row_list_new)

    def cal_cost_list(self):  # calculate the cost list using row list in populations
        if 0 != len(self.cost_list):
            self.cost_list = list()
        for i in self.population:
            cost = 0
            move_cost = self.move_cost(i)
            attack_cost = self.attack_cost(i)
            cost = move_cost + attack_cost
            self.cost_list.append(int(cost))

    def cal_prob(self,
                 cost_list):  # calculate the random selecting probability of each row list in population using its cost
        prob_list = list()
        temp = pd.DataFrame({'cost': cost_list})
        invert_cost_list = list(sum(temp['cost']) / temp['cost'])
        total_invert_cost = sum(invert_cost_list)
        for i in invert_cost_list:
            prob = i / total_invert_cost
            prob_list.append(prob)
        return prob_list

    def sort_cost_pop_list(self):  # sort population list according cost list in ascending order
        data = pd.DataFrame({"c": self.cost_list, "p": self.population})
        data_sorted = data.sort_values(by=['c'])
        cost_sorted = list(data_sorted['c'])
        population_sorted = list(data_sorted['p'])
        return (cost_sorted, population_sorted)

    def elitism(self):  # preserve best eli_num populations
        cost_sorted, population_sorted = self.sort_cost_pop_list()
        self.next_population_list.extend(population_sorted[:self.eli_num])

    def culling(self):  # delete worst cul_num populations
        cost_sorted, population_sorted = self.sort_cost_pop_list()
        cul_population = population_sorted[:len(population_sorted) - self.cul_num]
        cul_cost = cost_sorted[:len(population_sorted) - self.cul_num]
        return (cul_cost, cul_population)

    def get_parent(self, prob_list, population_list):  # randomly choose a population as parent
        prob = random.random()
        count = 0
        par = None
        for i in range(len(prob_list)):
            count += prob_list[i]
            if count >= prob:
                par = population_list[i]
                return par

    def cross_over(self, par1, par2):  # give two populations to generate two new populations
        child1 = list()
        child2 = list()
        child1 = par1[:self.crossover] + par2[self.crossover:]
        child2 = par2[:self.crossover] + par1[self.crossover:]
        return (child1, child2)

    def mutation(self, child):  # randomly mutation in child
        if self.mutate > 0:
            pro = random.random()
            if pro <= self.mutate:
                pos = random.randint(0, self.queen_num - 1)
                child = self.rand_change_pos(pos, child)

    def processing(self):  # mean part of genetic algorithm working
        if self.flag == 0:  # first time to process
            start_time = time.time()
            if 0 == len(self.population):
                # print("=====================Initializing===========================")
                self.init_weight_list()  # initial weight list
                self.init_environment()  # initial populations
                self.cal_cost_list()  # generate cost list for elistism and culling
                self.time_list.append(0)
            while self.generationNum < self.max_generation:
                self.elitism()  # elitism
                cul_cost, cul_population = self.culling()  # get cost and populations after culling
                self.min_cost_list.append(cul_cost[0])
                self.min_pop_list.append(cul_population[0])
                prob_list = self.cal_prob(cul_cost)  # calculate probability of current populations
                while len(
                        self.next_population_list) < self.pValue:  # while population not reach setted num, generate child and put it in list
                    par1 = self.get_parent(prob_list, cul_population)
                    par2 = self.get_parent(prob_list, cul_population)
                    child1, child2 = self.cross_over(par1, par2)
                    self.mutation(child1)
                    self.mutation(child2)
                    self.next_population_list.append(child1)
                    self.next_population_list.append(child2)
                self.population = self.next_population_list  # update population into next generation
                self.next_population_list = list()  # clean next population list
                self.cal_cost_list()
                self.generationNum += 1
                end_time = time.time()
                self.time_list.append(end_time - start_time)
                if self.max_time != 0:  # checking if time of running setted
                    if end_time - start_time >= self.max_time:  # check if have reach setted running time
                        break

            cost, pop = self.sort_cost_pop_list()
            self.min_cost_list.append(cost[0])
            self.min_pop_list.append(pop[0])
            self.min_cost = self.min_cost_list[-1]
            self.min_pop = self.min_pop_list[-1]
            self.process_time = end_time - start_time
            self.flag = 1
            # print('=====================Finishing===========================')
        else:  # for re-processing
            self.clean()
            self.processing()

    def clean(self):  # when finish, set augment to initial state, for user to run ag again.
        self.generationNum = 0
        self.population = list()
        self.cost_list = list()
        self.prob_list = list()
        self.min_cost_list = list()
        self.min_pop_list = list()
        self.flag = 0

    def get_board(self, row_list):  # form a numpy-array
        board = np.zeros((self.dim, self.dim)).astype(int)
        for i in range(len(self.init_col_list)):
            row = row_list[i]
            col = self.init_col_list[i]
            board[row][col] = self.weight_list[i]
        return board


# # initial a list of different dim board for testing
# def init_board_list(low_dim=8, up_dim=10,
#                     file_name=''):  # provide lower and upper boundary of dimension to generate a set of model and board, default lower = 5, upper = 10
#     model = None
#     board_list = []
#     for i in range(low_dim, up_dim + 1):
#         model = HeavyQueen9N8(chess_dim=i, file_name=file_name + 'test_{}.txt'.format(i, i))
#         board_list.append(model.init_board())
#     return board_list
#
#
# # load list of board txt in the path and form board list
# def load_txt(path):
#     file_list = glob.glob(path + '*.txt')
#     file_list.sort()
#     board_list = []
#     for i in file_list:
#         model = HeavyQueen9N8(load_file=i)
#         board_list.append(model.init_board())
#     return board_list


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
    N = 8 # can pick from 8-32
    file_name = f'board/test_{N}.txt'

    heavy_queen = HeavyQueen9N8(load_file=file_name)
    chess_board_init = heavy_queen.init_board()
    print("Original Board")
    print(chess_board_init)
    test_model = genetic_algo(chess_board_init)
    print("Initial Cost: {}".format(test_model.attack_cost(test_model.init_row_list)))
    start_time = time.time()
    test_model.processing()
    result = test_model.min_pop_list[0]
    board = test_model.get_board(result)
    print("Runtime:  %s seconds" % (time.time() - start_time))
    print("Result Board")
    # print(board)
    print("Result Cost: {}".format(test_model.min_cost_list[0]))

    ### draw chess board, please put it at the end of main
    plot_cub_size = int(800/N)
    draw_board = DrawBoard(value_list=board, is_plot=True, size=plot_cub_size)
    draw_board.drawchessboard()
