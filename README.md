# CS534_Assignments

# Part 1

#### Greedy Search

Go to the `CS534_Assignments (your cloned repository)`, run

```bash
python3 HWPart1_greedy.py
```

and input the number of queens N.

It may take long time if N is large. If you come across an issue that it takes long time with small N, please rerun the program.

#### A * Search

Run A*_and_Greedy_Haiyang.py. Type in the N as chessbord dimension.

If using Greedy Search: Change line 277 to : temp.cost_so_far=cal_f_greedy(temp.h_x,temp.g_x)

If using A* Search: Change line 277 to : temp.cost_so_far=cal_f_Astar(temp.h_x,temp.g_x)

# Part 2

#### Hill climbing

Run Hill climbing_final.py.

If use basic hill climbing: change line 208 to : h_model = hill_climbing(board, annealing = False)

If use hill climbing with annealing: change line 208 to : h_model = hill_climbing(board, annealing = True) 

we set a parameter T in hill_climbing class, it indicates the max temperature, default value is 400.

#### Genetic Algorithm

Run HW1Part2_genetic.py

If you want to modify the genetic algorithm model.

##### Change line 667: test_model = genetic_algo(chess_board_init, population=200, crossover=3, mutate=0.2, time=0, eli_ratio=0.2, cul_ratio=0.2, max_gen=1000), The parameters are listed with default value

population = the # of boards generate in each step

max_gen = the maximum # of generations

time = how many seconds to operate

eli_ratio = the percentage of boards we preserve in children list

eli_ratio = the percentage of boards we delete in parent list

mutate = probability of mutation

crossover = position to cut the board
