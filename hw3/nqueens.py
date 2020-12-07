import numpy as np
import copy
import random

## given a state of the board, return a list of all valid successor states
def succ(state, static_x, static_y):
    if state[static_x] != static_y:
        return []
    succ_list = []
    for i in range(0, len(state)):
        if i != static_x:
            if state[i] > 0:
                temp = copy.copy(state)
                temp[i] = temp[i] - 1
                succ_list.append(temp)
            if state[i] < (len(state)-1):
                temp = copy.copy(state)
                temp[i] = temp[i] + 1
                succ_list.append(temp)
    succ_list = sorted(succ_list)
    return succ_list            
    
## given a state of the board, return an integer score such that the goal state scores 0
def f(state):
    n = len(state)
    table = np.zeros(n,dtype = int)
    for i in range(0, n):
        if table[i] != 1:
            for j in range(0, n):
                # check if other queens exist in same row
                # check if other queens exist in diagnol
                if j!= i and ((abs(state[j] - state[i]) == abs(j - i))or state[j] == state[i]):
                    table[i] = 1
                    table[j] = 1               
    return sum(table)

## given the current state, use succ() to generate the successors and return the selected next state
def choose_next(curr, static_x, static_y):
    all_succ = succ(curr, static_x, static_y)
    if all_succ == []:
        return None
    all_succ.append(curr)
    # keep track the lowest f to find the optimal next state
    min_f = f(all_succ[0])
    possible = []
    for i in range(0, len(all_succ)):
        cur_f = f(all_succ[i])
        if cur_f < min_f:
            possible = []
            possible.append(all_succ[i])
            min_f = cur_f
        if cur_f == min_f:
            possible.append(all_succ[i])
    possible = sorted(possible)
    return possible[0]

## run the hill-climbing algorithm from a given initial state, return the convergence state
def n_queens(initial_state, static_x, static_y, print_path=True):
    curr = initial_state
    while 1:
        if print_path:
            print(curr, '- f=%d' %f(curr))
        # if f_score is zero, problem solved, return
        if f(curr) == 0:
            return curr
        # otherwise, choose next and check if state/f repeats
        next = choose_next(curr, static_x, static_y)
        if f(next) == f(curr):
            if print_path:
                print(next, '- f=%d' %f(next)) 
            return next
        curr = next
            
## run the hill-climbing algorithm on an n*n board with random restarts
def n_queens_restart(n, k, static_x, static_y):
    random.seed(1)
    result = []
    min_f = 100000000
    while k > 0:
        k = k - 1
        # randomly construct a new initial state 
        init = []
        for i in range (0, n):
            init.append(random.randint(0, n-1))
        # set static point
        init[static_x] = static_y
        # solve current problem
        ans = n_queens(init, static_x, static_y, False)
        # check if solution's f is zero
        f_ans = f(ans)
        if f_ans == 0:
            print(ans, '- f=%d' %f_ans)
            return
        # if not, if cur ans has smaller, update result list
        if f_ans < min_f:
            min_f = f_ans
            result = []
            result.append(ans)
        # if cur ans has same f, append to result list
        elif f_ans == min_f:
            result.append(ans)
        # otherwise, cur ans is not optimal sol, dont care
    # if reach k before get a solution with zero f value, print best solution 
    result = sorted(result)
    for i in range(0, len(result)):
        print(result[i], '- f=%d' %min_f)
    return
            