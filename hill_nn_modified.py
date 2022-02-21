import time;
import random
import numpy as np
from numpy import savetxt, loadtxt
queenPositions = []
count = 0


class HeavyQueen9N8: 
    def __init__(self, chess_dim=8, file_name='default.txt', load_file=None):
        self.dim = chess_dim
        self.file_name = file_name
        self.load_file = load_file
        self.chess_board = np.zeros((self.dim, self.dim)).astype(int)
        self.init_col_list = []
        self.init_row_list = []
        self.weight_list = []
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
        self.init_col_list = list(np.nonzero(load_mtx.T)[0])
        self.init_row_list = list(np.nonzero(load_mtx.T)[1])
        self.weight_list = self.init_weight_list()
        self.get_queen_pos()
   
    def init_weight_list(self): # initial the weight list
        weight_list = []
        for i in range(len(self.init_col_list)):
            weight_list.append(self.chess_board[self.init_row_list[i]][self.init_col_list[i]])
        return weight_list
    
    def get_queen_pos(self):
        for i in range(len(self.init_col_list)):
            pos = (self.init_row_list[i],self.init_col_list[i])
            self.queensPositions.append(pos)

    def test(self):
        print(self.n)
        pass



# test_model = HeavyQueen9N8(chess_dim=8, file_name='board_8.txt')




def findNextMove(curHVal,board, side_way = False):
   """
   This function will find next best move based on where
   heuristic value decreases sharply
   It can move any one of n queens
   """
   n = len(board)
   minHCalc = float('inf');     #Initializing to max     can we test our code now afer putting the weights?
   minPlayerMove = 0;      #Initializing
   minNewPosition = (0,0);
   
   for queen in range(n): #i.e. [1,...,n]
      #1. Finding all possible mover for this
      queenMoves = findPlacesToMove(queen);
      tempQueensPositionList = [x for x in queensPositions];
      for move in queenMoves: 
         #Change the position of this queen
         # initial_queen_pos = tempQueensPositionList[queen] 
         tempQueensPositionList[queen] = move; 
         weight = weight_list[queen]
         #Calculate the heuristic val
         hVal = heuristicValueOfPosition(tempQueensPositionList) + abs(queensPositions[queen][0] - move[0]) * pow(weight,2);
         #print("in min: hval is {} for {}".format(hVal, tempQueensPositionList))
         if (hVal < minHCalc):
            minHCalc = hVal;
            minPlayerMove = queen;
            minNewPosition = move;
         #print("\t\t new return vars are")   
   #print("hVal we are sending back is {}".format(minHCalc));
   if minHCalc < curHVal:
      return (minPlayerMove, minNewPosition, minHCalc)
   else:
       # pick a queen, randomly move it to another row
       # cost
        return -1; #That we have reached local minima

##################################################################
##################################################################

def heuristicValueOfPosition(tempQueenPosList):
   """
   Heuristic is no. of pairs of queens attacking each other
   in current positions scenario
   """
   hVal = 0;
   
   for q1 in range(0,len(tempQueenPosList)): #i.e. [1,...,n-1]
      q1X = tempQueenPosList[q1][0];
      q1Y = tempQueenPosList[q1][1];
      for q2 in range(q1+1,len(tempQueenPosList)): #i.e. [i,i+1, ....., n]
         q2X = tempQueenPosList[q2][0];
         q2Y = tempQueenPosList[q2][1];
         if (q1X == q2X) or (q1Y == q2Y) or (abs(q1X-q2X) == abs(q1Y-q2Y)):
            #then q1 and q2 are not attacking each other
            hVal += 1;
            #print("hVal is {} since {} <-> {}".format(hVal, q1, q2))
   return hVal * 100;
      
##################################################################
##################################################################
      
def findPlacesToMove(queenNumber):
   """
   #This function will find location of all the plasces a queen
   can move to.
   """
   n = len(board)
   movesDestinations = [];
   currentQueenPos = queensPositions[queenNumber];
   #1.Search horizontally:
      # Searching in +ve direction
   for x in range(1,n): #i.e 1,2,3
     if (currentQueenPos[0]+x < n) and ((currentQueenPos[0]+x, currentQueenPos[1]) not in queensPositions):
         movesDestinations.append((currentQueenPos[0]+x, currentQueenPos[1]));
     else:
         break;

      # Searching in -ve direction
   for x in range(1,n): #i.e -1,-2,-3
      if (currentQueenPos[0]-x >= 0) and ((currentQueenPos[0]-x, currentQueenPos[1])) not in queensPositions:
         movesDestinations.append((currentQueenPos[0]-x, currentQueenPos[1]));
      else:
         break;

   #2.Search vertically:
      # Searching in +ve direction
   #for y in range(1,n): #i.e 1,2,3
      #if (currentQueenPos[1]+y <= n) and ((currentQueenPos[0], currentQueenPos[1]+y) not in queensPositions):
         #movesDestinations.append((currentQueenPos[0], currentQueenPos[1]+y));
      #else:
        # break;

      # Searching in -ve direction
   #for y in range(1,n): #i.e -1,-2,-3
     # if (currentQueenPos[1]-y >= 1) and ((currentQueenPos[0], currentQueenPos[1]-y)) not in queensPositions:
         #movesDestinations.append((currentQueenPos[0], currentQueenPos[1]-y));
      #else:
         #break;



   return movesDestinations;

##################################################################
##################################################################
def ai(board,queensPositions,weight_list):
   start_time = time.time()
   time_list = []
   cost_list = []
   curHVal = heuristicValueOfPosition(queensPositions);
   queensPositions_final = queensPositions[:]
   while True:
      time.sleep(1);
      queenNewPos = findNextMove(curHVal,board); #It returns (queen, new pos, hVal)
      if(queenNewPos == -1):
         print("Reached global minima, cant make more moves !");
         break;
      queensPositions_final[queenNewPos[0]] = queenNewPos[1];
      curHVal = queenNewPos[2];
      print("Current Board position with {} is : {}".format(curHVal,queensPositions_final));
      end_time = time.time()
      time_list.append(end_time-start_time)
      cost_list.append(curHVal)
   print("Final positionsa are {}".format(queensPositions_final));
   return (time_list, cost_list)

if __name__ == "__main__":
    test_model = HeavyQueen9N8(load_file='test_8.txt')
    board = test_model.init_board()
    queensPositions = test_model.queensPositions;
    weight_list = test_model.weight_list
    print(queensPositions)
    print(weight_list)
    time_list, cost_list = ai(board,queensPositions,weight_list)
    print(time_list)
    print(cost_list)
    print(queensPositions)