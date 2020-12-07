import numpy as np
import copy
import heapq

## This function computes h value based on a given puzzle
def comp_h(state):
    h = 0
    for i in range(0,9):
        if state[i] != 0:
            expect_r = int(i/3)
            expect_c = i%3
            cur_r = int((state[i]-1)/3)
            cur_c = (state[i]-1)%3
            d = abs(expect_r - cur_r) + abs(expect_c - cur_c)
            h += d
    return h

## This function generates all possible moving conditions
def generate_succ(state):    
    # first construct given puzzle and find the position of 0
    puzzle = np.array([state[0:3],state[3:6],state[6:9]])
    row = 0
    col = 0
    for i in range(0,3):
        for j in range(0,3):
            if puzzle[i][j] == 0:
                row = i
                col = j
    
    # now determine the four possible moves
    # and save all possible succ state into a list
    num_pos_move = 0
    output = []
    # move 0 up
    if row != 0:
        num_pos_move += 1
        up = copy.copy(puzzle)
        up[row][col] = up [row - 1][col]
        up[row - 1][col] = 0
        temp = []
        for i in range(0,3):
            for j in range(0,3):
                temp.append(up[i][j])
        output.append(temp)
    # move 0 down
    if row != 2:
        num_pos_move += 1
        down = copy.copy(puzzle)
        down[row][col] = down[row + 1][col]
        down[row + 1][col] = 0
        temp = []
        for i in range(0,3):
            for j in range(0,3):
                temp.append(down[i][j])
        output.append(temp)
    # move 0 to the left
    if col != 0:
        num_pos_move += 1
        left = copy.copy(puzzle)
        left[row][col] = left[row][col - 1]
        left[row][col - 1] = 0
        temp = []
        for i in range(0,3):
            for j in range(0,3):
                temp.append(left[i][j])
        output.append(temp)
    # move 0 to the right
    if col != 2:
        num_pos_move += 1
        right = copy.copy(puzzle)
        right[row][col] = right[row][col + 1]
        right[row][col + 1] = 0
        temp = []
        for i in range(0,3):
            for j in range(0,3):
                temp.append(right[i][j])
        output.append(temp)
    # now, we have to sort the output list
    output = sorted(output)
    return output, num_pos_move

## This function prints out all possible moving conditions
def print_succ(state):    
    output, num_pos_move = generate_succ(state)    
    # finally, compute h for each condition and print it out
    for i in range(0,num_pos_move):
        h = comp_h(output[i])
        print(output[i], "h=%d" %h)
                
## This function search as a certain parent
def search_parent(par_num, CLOSED):
    for i in range (0, len(CLOSED)):
        if CLOSED[i][2][2]+1 == par_num:
            break
    return CLOSED[i]
    
## This function prints the solution of the puzzle
def print_sol(item, CLOSED):
    moves = 0
    cur = copy.copy(item)
    if cur[2][2] != -1:
        moves = print_sol(search_parent(cur[2][2],CLOSED),CLOSED)
    print(cur[1], 'h=%d' %comp_h(cur[1]), 'moves:', moves)
    return moves + 1


## This function solves the puzzle
def solve(state):
    # step 1: Put the start state on the priority queue, called OPEN
    OPEN = []
    CLOSED = []
    h = comp_h(state)
    g = 0
    ''' Original author: CS540 Instructors and TAs
       Source: Canvas HW2 page
       The following line that push an item into priority queue and pop from the queue
    '''
    # note that f(n)=g(n)+h(n) and parent index of -1 denotes initial state
    heapq.heappush(OPEN, (g+h, state, (g, h, -1)))
    while 1: 
        # Since it is safe to assume the input puzzle is always solvable
        # Then I will skip step two that checks empty and fail if empty
        # Step 3: Remove from OPEN and place on CLOSED a node n for which f(n) is minimum
        MIN = heapq.heappop(OPEN)
        heapq.heapify(OPEN)
        CLOSED.append(MIN)
        # Step 4:If goal, exit
        if MIN[2][1] == 0:
            print_sol(MIN,CLOSED)
            print
            return 
        # Step 5:generate all succ
        g_nxt = MIN[2][0] + 1
        states_nxt, num_pos_state = generate_succ(MIN[1])
        for i in range(0, num_pos_state):
            found = False
            update = False
            h_nxt = comp_h(states_nxt[i])
            for j in range(0, len(CLOSED)):
                if CLOSED[j][1] == states_nxt[i]:
                    found = True
                    if CLOSED[j][2][0] > g_nxt:
                        update = True
                        #CLOSED.pop(j)
                    break
            for j in range(0, len(OPEN)):
                if OPEN[j][1] == states_nxt[i]:
                    found = True
                    if OPEN[j][2][0] > g_nxt:
                        OPEN.pop(j)
                        heapq.heapify(OPEN)
                        update= True
                    break
            if found == False:
                heapq.heappush(OPEN,(g_nxt+h_nxt, states_nxt[i], (g_nxt,h_nxt,MIN[2][2]+1)))
            elif update == True:
                heapq.heappush(OPEN,(g_nxt+h_nxt,states_nxt[i],(g_nxt,h_nxt,MIN[2][2]+1)))

    return                                                                 