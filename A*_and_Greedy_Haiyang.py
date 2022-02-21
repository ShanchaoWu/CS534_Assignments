########################################################################################
# Author: Haiyang Yun

# Node is basic class for chess board. It contains h_value, g_value, f_value, and metadata

# is_Goal(state): used to check if current chessboard has reached 0 h_value, which means no attacking pairs

# random_state(N): Generate randow N by N chessboard. Every Queen has weighy 1~9

# cal_heruistic(state): Compute current chessboard's h_value.

# cal_g(state1, state2): Compute the movement cost between two states. Cost function  = weight^2 * steps

# cal_f_greedy(h_value,g_value), cal_f_Astar(h_value,g_value): Different approaches to guide the search

# populate(Node): Generate every possible children state given the parent Node. Compute their h, g, and f. 
# Then append children in pq_list, store f value and pq_list index into PriorityQueue

# main(): Read user input as N, generate start node. Use populate(x) to generate children. Then choose the child with lowest f_value as next node. 
# Break and return if Is_Goal(stete) return a True. Print parents nodes of the solution
########################################################################################



import numpy as np
import time
from queue import PriorityQueue
import random
import math
global pq
import copy
pq=PriorityQueue()
global pq_list
pq_list=[] # open list

class Node:
    def __init__(self):
        self.state=np.zeros((N,N))
        self.g_x=0
        self.h_x=0
        self.cost_so_far=0
        self.parent= None
        #self.f_x = 0
        pass


def is_goal(state):
    #checking same row
    #checking same column
    #checking diagonal1
    #checking diagonal
    #checking row on left side
    x=np.where(state != 0)
    N=len(state)
    for i in range (len(x[0])):
        row=x[0][i]
        col=x[1][i]
        
        count=0
        for i in range(N):
            if(state[row][i]!=0):
                count=count+1
        
        if count>1:
            return 0
        
        count=0
        for i in range(N):
            if(state[i][col]!=0):
                count=count+1
        
        if count>1:
            return 0
        
        count=0
       
        #Primary Diagonal 
        row1=row-1
        col1=col-1
        
        while row1>=0 and row1<N and col1>=0 and col1<N:
           
            if(state[row1][col1]!=0):
                return 0
            row1=row1-1
            col1=col1-1
        
        row1=row+1
        col1=col+1
        
        while row1>=0 and row1<N and col1>=0 and col1<N:
            
            if(state[row1][col1]!=0):
                return 0
            row1=row1+1
            col1=col1+1
            
        row1=row-1
        col1=col+1
        
        #Secondary diagonal
        while row1>=0 and row1<N and col1>=0 and col1<N:
            
            if(state[row1][col1]!=0):
                return 0
            row1=row1-1
            col1=col1+1
            
        row1=row+1
        col1=col-1
        
        while row1>=0 and row1<N and col1>=0 and col1<N:
            
            if(state[row1][col1]!=0):
                return 0
            row1=row1+1
            col1=col1-1
            
    return 1

def random_state(N):
    board = np.zeros((N,N)).astype(int)
    print("Creating Random First State for N =",N)
    for x in range(N):
        row=random.randrange(0,N)
        #print(row,x)
        board[row][x]=random.randint(1,9)
        #board[row][x]=4
    return board

def cal_heuristic(state):
    N=len(state)
    x=np.where(state != 0)
    attack=[]
    count=0
    for i in range (len(x[0])):
        row=x[0][i]
        col=x[1][i]
        
        
        for i in range(N):
            if(state[row][i]!=0 and i!=col):
                count=count+1
                attack.append([[row,col],[row,i]])
        
        for i in range(N):
            if(state[i][col]!=0 and i!=row):
                count=count+1
                attack.append([[row,col],[row,i]])
        
        

        
        #Primary diagonal - upper left
        row1=row-1
        col1=col-1
        
        while row1>=0 and row1<N and col1>=0 and col1<N:
            
            if(state[row1][col1]!=0):
                count=count+1
                #print("Same primary diagonal1",count)
                attack.append([[row,col],[row1,col1]])
            row1=row1-1
            col1=col1-1
        
        #lower right
        row1=row+1
        col1=col+1
        
        while row1>=0 and row1<N and col1>=0 and col1<N:
            
            if(state[row1][col1]!=0):
                count=count+1
                #print("Same primary diagonal2",count)
                attack.append([[row,col],[row1,col1]])
        
            row1=row1+1
            col1=col1+1
            
        row1=row-1
        col1=col+1
        
        #Secondary diagonal - upper right
        while row1>=0 and row1<N and col1>=0 and col1<N:
            
            if(state[row1][col1]!=0):
                count=count+1
                #print("Same secondary diag",count)
                attack.append([[row,col],[row1,col1]])
        
            row1=row1-1
            col1=col1+1
            
        row1=row+1
        col1=col-1
        
        #lower left
        while row1>=0 and row1<N and col1>=0 and col1<N:
            
            if(state[row1][col1]!=0):
                count=count+1
                #print("Same secondary diag",count)
                attack.append([[row,col],[row1,col1]])
        
            row1=row1+1
            col1=col1-1
    #print("Cal Heuristic",count/2)
    #print(attack)
    temp=math.floor((count/2))
    if temp==0:
        return 0
    return (math.floor((count/2)))

def cal_g(state1,state2):
    cost = 0
    if (np.array_equal(state1,state2)==0):
        changed_state=np.absolute(state1-state2)
        
        ones=np.where(changed_state!=0)[0] # row number
        ones2=np.where(changed_state!=0)[1] # col number
        # according to col number, can track the weight of queens
        cols = np.unique(ones2)
        for i in cols:
            pos = np.where(ones2 == i)[0]
            row_diff = abs(ones[pos[1]] - ones[pos[0]])
            if state1[ones[pos[1]],i] != 0:
                weight = state1[ones[pos[1]],i]
            elif state2[ones[pos[1]],i] != 0:
                weight = state2[ones[pos[1]],i]
            else:
                print("wrong in finding weight")
                return 0
            # print(weight)
            cost = cost + np.square(weight)*row_diff
         
        return cost
    else:
        return 0

def cal_f_greedy(h_value,g_value):
    return h_value

def cal_f_Astar(h_value,g_value):
    return h_value + g_value

def populate(x):
    global pq
    global pq_list
    a=None
    temp=None
    state=np.copy(x.state)
    l_weight = np.amin(state)
    for col in range(len(state)):
        state=np.copy(x.state)
        for i in range(len(state)):
            if state[i][col] != 0:
                cur_weight = state[i][col]
                #print(i,col,cur_weight)
        for row in range(len(state)):
            temp=None
            
            temp=Node()
            temp.parent=x
            
            #state[col][:]=0
            for k in range(len(state)):
                state[k][col]=0
            state[row][col] = cur_weight
            temp.state=state
            temp.h_x=cal_heuristic(temp.state)
            temp.g_x=cal_g(temp.state,x.state)+ x.g_x
            #print (temp.g_x)
            #print (temp.h_x)
            #temp.cost_so_far=temp.h_x
            #temp.cost_so_far=temp.h_x+temp.g_x
            temp.cost_so_far=cal_f_Astar(temp.h_x,temp.g_x)
            #temp.cost_so_far=cal_f_greedy(temp.h_x,temp.g_x)

            
            #print (temp)
            #print (temp.cost_so_far)
            #print("I am printing in populate")
            #print (temp.state)
            #debug = np.array_equal(x.state,temp.state)
            if np.array_equal(x.state,temp.state)==False:
                a=copy.deepcopy(temp)
                pq_list.append(a)
                #print(temp.cost_so_far)
                #print(len(pq_list)-1)
                
                #pq.put((temp.cost_so_far,len(pq_list)-1))
                pq.put((temp.cost_so_far,len(pq_list)-1))

def print_soln(state1):
    list_soln=[]
    n= len(state1.state)
    while(state1.parent!=None):
        list_soln.append(state1.state)
        state1=state1.parent
    print ("Branching Factor",n*(n-1))
    while(len(list_soln)!=0):
        #print("Printing soln")
        a=list_soln.pop()
        print(a)


print("Enter N")
N=int(input())
time_start=time.perf_counter()
start_state=random_state(N)
print("INITIAL START STATE")
print(start_state)
start=Node()
start.state=start_state
while(True):
    populate(start)
    next_indice=pq.get()
    #print("Indice",next_indice[1])
    next_state=pq_list[next_indice[1]]
    #print("Next STate")
    #print (next_indice[1])
    # print(next_state.state)
    # print(next_state.h_x)
    # print(next_state.g_x)
    # print(next_state.cost_so_far)
    goal = is_goal(next_state.state)
    if(goal==1):
        time_end=time.perf_counter()
        print("Solution reached")
        print_soln(next_state)
        print("Number of Nodes expanded",len(pq_list))
        print ("Effective Cost to solve the problem",next_state.cost_so_far)
        print ("Total Time Taken",time_end-time_start)
        break
    
    start=next_state
