import time;
import random
import numpy as np
from numpy import savetxt, loadtxt
import math

class HeavyQueen9N8: 
    def __init__(self, chess_dim=8, file_name='default.txt', load_file=None):
        self.dim = chess_dim
        self.file_name = file_name
        self.load_file = load_file
        self.chess_board = np.zeros((self.dim, self.dim)).astype(int)
        self.queensPositions = []
        

    def init_board(self):
        if self.load_file == None:
            for i in range(self.dim):
                self.chess_board[random.randint(0, self.dim-1), i] = random.randint(1, 9)
            flag = int(self.dim/8)
            while(flag):
                col = random.randint(0, self.dim-1)
                row = random.randint(0, self.dim-1)
                while self.chess_board[row, col] != 0:
                    col = random.randint(0, self.dim-1)
                    row = random.randint(0, self.dim-1)
                self.chess_board[row, col] = random.randint(1,9)
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

    def test(self):
        print(self.n)
        pass

    
    
    
class hill_climbing:
    
    def __init__(self, board, annealing = False, T = 500):
        self.init_board = board
        self.dim = len(board)
        self.weight_list = self.init_weight_list()
        self.init_queen_pos = self.get_queen_pos()
        self.annealing_flag = annealing
        self.chess_num = len(self.weight_list)
        self.queensPositions_final = self.init_queen_pos[:]
        self.max_t = T
        
    def init_weight_list(self):  # initial the weight list
        weight_list = []
        init_col_list = list(np.nonzero(self.init_board.T)[0])
        init_row_list = list(np.nonzero(self.init_board.T)[1])
        for i in range(len(init_row_list)):
            weight_list.append(self.init_board[init_row_list[i]][init_col_list[i]])
        return weight_list
    
    def get_queen_pos(self): # get the queen position as list of (row, col)
        queensPositions = []
        init_col_list = list(np.nonzero(self.init_board.T)[0])
        init_row_list = list(np.nonzero(self.init_board.T)[1])
        for i in range(len(init_col_list)):
            pos = (init_row_list[i], init_col_list[i])
            queensPositions.append(pos)
        return queensPositions
    
    def get_attack_cost(self, tempQueenPosList):
        hVal = 0
        for q1 in range(len(tempQueenPosList)): 
            q1X = tempQueenPosList[q1][0]
            q1Y = tempQueenPosList[q1][1]
            for q2 in range(q1+1,len(tempQueenPosList)):
                q2X = tempQueenPosList[q2][0]
                q2Y = tempQueenPosList[q2][1]
                if (q1X == q2X) or (q1Y == q2Y) or (abs(q1X-q2X) == abs(q1Y-q2Y)):
                    hVal += 1
        return hVal * 100
    
    def get_move_cost(self, tempQueenPosList):
        hVal = 0
        for i in range(len(self.init_queen_pos)):
            weight = self.weight_list[i]
            hVal = hVal + abs(self.init_queen_pos[i][0] - tempQueenPosList[i][0]) * pow(weight, 2)
        return hVal
    
    
    def findNextMove(self, curHVal):
        minHCalc = float('inf')
        minPlayerMove = 0
        minNewPosition = (0,0)
        move_candidate = []
        for queen in range(len(self.init_queen_pos)): 
            queenMoves = self.findPlacesToMove(queen);
            tempQueensPositionList = [x for x in self.queensPositions_final]
            for move in queenMoves: 
                tempQueensPositionList[queen] = move;
                hVal = self.get_attack_cost(tempQueensPositionList) + self.get_move_cost(tempQueensPositionList)
                if hVal < minHCalc:
                    if len(move_candidate) == 0:
                        minHCalc = hVal
                        minPlayerMove = queen
                        minNewPosition = move
                        move_candidate.append((hVal, queen, move))
                    else:
                        move_candidate = []
                        minHCalc = hVal
                        minPlayerMove = queen
                        minNewPosition = move
                        move_candidate.append((hVal, queen, move))
                elif hVal == minHCalc:
                    move_candidate.append((hVal, queen, move))
        if minHCalc < curHVal:
            c = random.randint(0, len(move_candidate)-1)
            minHCalc = move_candidate[c][0]
            minPlayerMove = move_candidate[c][1]
            minNewPosition = move_candidate[c][2]
            return(minPlayerMove, minNewPosition, minHCalc)
        else:
            return -1
      
    def findPlacesToMove(self, queenNumber):
        movesDestinations = []
        currentQueenPos = self.init_queen_pos[queenNumber]
        for x in range(1, self.dim): #i.e 1,2,3
            if (currentQueenPos[0]+x < self.dim):
                if ((currentQueenPos[0]+x, currentQueenPos[1]) not in self.init_queen_pos):
                    movesDestinations.append((currentQueenPos[0]+x, currentQueenPos[1]))
            else:
                break
        for x in range(1, self.dim): #i.e -1,-2,-3
            if (currentQueenPos[0]-x >= 0):
                if ((currentQueenPos[0]-x, currentQueenPos[1]) not in self.init_queen_pos):
                    movesDestinations.append((currentQueenPos[0]-x, currentQueenPos[1]))
            else:
                break
        return movesDestinations

    def processing(self):
        start_time = time.time()
        curHVal = self.get_attack_cost(self.init_queen_pos);
        time_list = [0]
        cost_list = [curHVal]
        success = 0
        while True:
            time.sleep(1);
            queenNewPos = self.findNextMove(curHVal); #It returns (queen, new pos, hVal)
            if(queenNewPos == -1):
                if self.annealing_flag != 0:
                    print("using annealing")
                    q = random.randint(0, self.chess_num-1)
                    pos = self.queensPositions_final[q]
                    side_way = self.queensPositions_final[:]
                    while True:
                        move = (random.randint(0, self.dim), pos[1])
                        if move != pos and move not in side_way:
                            break
                    side_way[q] = move
                    hVal_an = self.get_attack_cost(side_way) + self.get_move_cost(side_way)
                    rand = random.random()
                    delta = hVal_an - curHVal
                    prob = math.exp(-delta / self.max_t)
                    if rand < prob:
                        success = 1
                    else:
                        break
                else:
                    print("Reached global minima, cant make more moves !")
                    break
            if success == 0:
                self.queensPositions_final[queenNewPos[0]] = queenNewPos[1]
                curHVal = queenNewPos[2];
                print("Current Board position with {} is : {}".format(curHVal, self.queensPositions_final));
            else:
                self.queensPositions_final = side_way
                curHVal = hVal_an
                success = 0
            end_time = time.time()
            time_list.append(end_time - start_time)
            cost_list.append(curHVal)
        print("Final positionsa are {}".format(self.queensPositions_final));
        return (time_list, cost_list)
    
    def get_board(self):  # form a numpy-array
        board = np.zeros((self.dim, self.dim)).astype(int)
        i = 0
        for pos in self.queensPositions_final:
            row = pos[0]
            col = pos[1]
            board[row][col] = self.weight_list[i]
            i += 1
        return board
        

if __name__ == "__main__":
    
    test_model = HeavyQueen9N8(load_file='test_8.txt')
    board = test_model.init_board()
    print(board)
    h_model = hill_climbing(board, annealing = True) # annealing = True means using annealing, False means hill climbing. T = 400 default, can modeify, it is the max temperature. large T makes high possible moving side-way
    time_list, cost_list = h_model.processing()
    print(h_model.get_board())
    print(time_list)
    print(cost_list)