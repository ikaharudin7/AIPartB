# Evaluation function, takes the board at a given time before a move and evaluates how good that board is. 
from collections import defaultdict
from math import inf
import math
from tkinter.messagebox import YES

STEP_SIZE = 1

# Try this minimax function for pruning
def minimax(size, state, player_colour, maximising_player, alpha, beta, depth):
    if player_colour == "red":
        opponent = "blue"
    else:
        opponent = "red"

    # check if game is over
    if depth >= 3:
        eval = evaluation(player_colour, state, size)
        return (eval, state)
    
    if (maximising_player):
        maxEval = -math.inf
        maxState = [] # might be error here
        # go through each child of current state
        successors = successor(state, size, player_colour)
        for s in successors[0]: 
            item = minimax(size, s, opponent, 0, alpha, beta, depth+1)
            print("maximising", item)
            eval = item[0]
            maxEval = max(maxEval, eval) 
            maxState = s # might be error here with order
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
            return (maxEval, maxState)

    else:
        minEval = math.inf
        minState = [] # might be error here
        successors = successor(state, size, player_colour)
        for s in successors[0]: 
            item = minimax(size, s, opponent, 1, alpha, beta, depth+1)
            print("minimising", item)
            eval = item[0]
            minEval = min(minEval, eval)
            minState = s
            beta = min(beta, eval)
            if beta <= alpha:
                break
            return (minEval, minState)




# Gets the successor states from current state
def successor(board, size, colour):
    
    states = []
    new_coords = []

    for i in range(size):
        for j in range(size):

            current = []
            for coordinate in board:
                current.append(coordinate)

            if (i, j, "blue") not in board and (i, j, "red") not in board:
                coord = (i, j, colour)
                # Use the capture function when adding in 
                disposable = capture(current, coord)
                #print("captured ", disposable)
                # Add coord to current board
                current.append(coord)
                #print("successor state ", current)
                # Remove captured elements
                if len(disposable) > 0:
                    for item in disposable:
                        current.remove(item)
                
                states.append(current)
                new_coords.append((i, j, colour))

    return [states, new_coords]



def max_val(size, player, state, alpha, beta, alpha_state, beta_state, depth):
    
    if player == "red":
        opponent = "blue"
    else:
        opponent = "red"

    if depth >=3:
        eval_value = evaluation(player, state, size)
        return (eval_value, state)

    successors = successor(state, size, player)
    for s in successors[0]: 
        item = min_val(size, opponent, s, alpha, beta, alpha_state, beta_state, depth + 1)
        #print("successor eval", item[0], "alpha", alpha)
        if alpha < item[0]:
            alpha_state = s
        alpha = max(alpha, item[0])

        if alpha >= beta:
            return (beta, beta_state)
        
    
    return (alpha, alpha_state)


def min_val(size, player, state, alpha, beta, alpha_state, beta_state, depth):
    
    if player == "red":
        opponent = "blue"
    else:
        opponent = "red"

    if depth >=3:
        eval_value = evaluation(player, state, size)
        #print("deepest", eval_value)
        return (eval_value, state)
    
    successors = successor(state, size, player)
    for s in successors[0]: 
        item = max_val(size, opponent, s, alpha, beta, alpha_state, beta_state, depth + 1)
        if beta > item[0]:
            beta_state = s
        beta = min(beta, item[0])

        if beta <= alpha: 
            return (alpha, alpha_state)
        
    
    return (beta, beta_state)


def evalfunc(n, board, vertical, colour):

    # OFFENCE - add values which fill up along the AStar path
    # (Give more value to a state which has 2 more hexes to go compared to 4 more to win)
    # Use A-Star initially to go from start to end
    middle = int(n / 2)
    # Have a check if the end goals are taken or not

    if vertical:
        
        choose = aStar((n - 1, middle), (0, middle), n, board, colour)

    else:
        choose = aStar((middle - 2, n - 1), (middle, 0), n, board, colour)


    for tup in choose:
        # Problem here as self.board has different format. 
        if (tup[0], tup[1], "red") not in board and (tup[0], tup[1], "blue") not in board:
            cs = tup
            break
    
    # Replace 
    return ("PLACE", cs[0], cs[1])

    # DEFENCE - add greatest value to a block/capture if opp is 1 or 2 away from winning

    # Find a different AStar Path/ Block

def evaluation(player, state, size):
    if player == "red":
        opponent = "blue"
    else:
        opponent = "red"

    # Weightings for each type of board available. 
    winning_weight = 5

    curr_team = close_to_win(player, state, size)[2] * winning_weight
    opp_team = close_to_win(opponent, state, size)[2] * winning_weight

    return curr_team - opp_team

# For a given player, check if a branch of the current hex reaches the other side of the board
# Return 1 if it has arrived on the other side, 0 if the last hex on the branch is not
def reach_other_side(board, size, player, curr, blocked):
    if (player == "red"):
        if (curr[0] == size - 1): # current hex is at the other side
            return 1
        neighbours = generate_branch(curr, size, blocked)
        for coord in neighbours:
            if (reach_other_side(board, size, player, coord, blocked) == 1):
                return 1
        return 0
            
    else:
        if (curr[1] == size - 1): # current hex is at the other side
            return 1
        neighbours = generate_branch(curr, size, blocked)
        for coord in neighbours:
            if (reach_other_side(board, size, player, coord, blocked) == 1):
                return 1
        return 0


# Returns the string of the player which has already won, or if no player has yet won, "none"
def which_player_won(board, size):
    # check whether the red player has won
    red_reach_other_side = 0
    for q in range(size):
        if (player_in_coord(board, (0, q), "red")):
            # The opponent is the blocked hexes
            blocked = []
            for hex in board:
                if (hex[2] != "red"):
                    blocked.append((hex[0], hex[1]))
            # check if a branch of this hex reaches the other side of the board
            if (reach_other_side(board, size, "red", (0, q), blocked) == 1):
                red_reach_other_side = 1

    # check whether the red player has won
    blue_reach_other_side = 0
    for r in range(size):
        if (player_in_coord(board, (r, 0), "blue")):
            # The opponent is the blocked hexes
            blocked = []
            for hex in board:
                if (hex[2] != "blue"):
                    blocked.append((hex[0], hex[1]))
            # check if a branch of this hex reaches the other side of the board
            if (reach_other_side(board, size, "blue", (0, q), blocked) == 1):
                blue_reach_other_side = 1

    
    if (red_reach_other_side):
        return "red wins"
    elif (blue_reach_other_side): 
        return "blue wins"
    else:
        return "none"



# Detects how close a player is to winning
def close_to_win(player, board, size):

    min_steps = inf
    ideal_path = []
    remaining_steps = []
    # return the ideal path, remaining steps, and number of these steps
    return_item = []  

    if (player == "red"):
        for i in range(size): 
            # selects a starting position in first line
            if (player_in_coord(board, (0, i), "blue") != 1): 
                start = (0, i)
                # selects the end/goal position in final line
                for j in range(size):
                    if (player_in_coord(board, (size - 1, j), "blue") != 1):
                        end = (size - 1, j) 
                        # find the shortest path between the set points
                        path = aStar(start, end, size, board, player)
                        # check that there is a path
                        if (path != 0):
                            path_without_red = path_without_player(board, path, player)
                            # update the minimum number of steps needed to get to win
                            if (len(path_without_red) < min_steps):
                                min_steps = len(path_without_red)
                                ideal_path = path
                                remaining_steps = path_without_red

    else: # this is a blue player
        for i in range(size): 
            if (player_in_coord(board, (i, 0), "red") != 1): 
                start = (i, 0)
                for j in range(size):
                    if (player_in_coord(board, (j, size - 1), "red") != 1):
                        end = (j, size - 1) 
                        path = aStar(start, end, size, board, player)
                        if (path != 0):
                            path_without_blue = path_without_player(board, path, player)
                            if (len(path_without_blue) < min_steps):
                                min_steps = len(path_without_blue)
                                ideal_path = path
                                remaining_steps = path_without_blue

    return_item.append(ideal_path)
    return_item.append(remaining_steps)
    return_item.append(min_steps)

    return return_item


# Returns a list of hexes in a given path, which does not contain player
def path_without_player(board, path, player):
    remaining_hexes = []
    for coord in path:
        if (player_in_coord(board, coord, player)):
            continue
        else:
            remaining_hexes.append(hex)
    
    return remaining_hexes


# Determines whether board contains a player located at given coordinate
def player_in_coord(board, coord, player):
    return_val = 0
    for hex in board:
        if (hex == (coord[0], coord[1], player)):
            return_val = 1
    
    return return_val


# Detecting capture and returning captured elements - coord in (r, q, colour) format
def capture(board, coord):
    colour = coord[2]
    opp = "red"
    if colour == "red":
        opp = "blue"

    captured = []

    # Check 6 possible options manually
    r = coord[0]
    q = coord[1]

    # Check if it is on the upper and lower segments of diamond
    if (r+2, q-1, colour) in board and (r+1, q-1, opp) in board and (r+1, q, opp) in board:
        captured.append((r+1, q-1, opp)) 
        captured.append((r+1, q, opp))
    if (r+1, q+1, colour) in board and (r+1, q, opp) in board and (r, q+1, opp) in board:
        captured.append((r+1, q, opp))
        captured.append((r, q+1, opp))
    if (r-1, q+2, colour) in board and (r, q+1, opp) in board and (r-1, q+1, opp) in board:
        captured.append((r, q+1, opp))
        captured.append((r-1, q+1, opp))
    if (r-2, q+1, colour) in board and (r-1, q, opp) in board and (r-1, q+1, opp) in board:
        captured.append((r-1, q, opp))
        captured.append((r-1, q+1, opp))
    if (r-1, q-1, colour) in board and (r, q-1, opp) in board and (r-1, q, opp) in board:
        captured.append((r, q-1, opp))
        captured.append((r-1, q, opp))
    if (r+1, q-2, colour) in board and (r, q-1, opp) in board and (r+1, q-1, opp) in board:
        captured.append((r, q-1, opp))
        captured.append((r+1, q-1, opp))

    # Check if inner two have been completed to capture upper segments 
    if (r, q+1, colour) in board and (r+1, q, opp) in board and (r-1, q+1, opp) in board:
        captured.append((r+1, q, opp)) 
        captured.append((r-1, q+1, opp))
    if (r-1, q+1, colour) in board and (r, q+1, opp) in board and (r-1, q, opp) in board:
        captured.append((r, q+1, opp))
        captured.append((r-1, q, opp))
    if (r-1, q, colour) in board and (r, q-1, opp) in board and (r-1, q+1, opp) in board:
        captured.append((r, q-1, opp))
        captured.append((r-1, q+1, opp))
    if (r, q-1, colour) in board and (r+1, q-1, opp) in board and (r-1, q, opp) in board:
        captured.append((r+1, q-1, opp))
        captured.append((r-1, q, opp))
    if (r+1, q-1, colour) in board and (r, q-1, opp) in board and (r+1, q, opp) in board:
        captured.append((r, q-1, opp))
        captured.append((r+1, q, opp))
    if (r+1, q, colour) in board and (r, q+1, opp) in board and (r+1, q-1, opp) in board:
        captured.append((r, q+1, opp))
        captured.append((r+1, q-1, opp))
    

    return unique(captured)



# Change an array such that it only has unique values
def unique(list):
 
    unique_list = []
     
    for x in list:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    
    return unique_list



# A star algorithm
def aStar(start, goal, size, board, colour):

    # Keep track of which hexes have been blocked
    blocked = []
    for hex in board:
        if hex[2] != colour:
            blocked.append((hex[0], hex[1]))
    start = (start[0], start[1])
    goal = (goal[0], goal[1])
    traversed = []

    # Define set of hexagons who's children have not been fully traversed
    open = set()
    open.add(start)

    descended_from = {}
    
    # Dictionaries defining a node and its gn and fn values
    g_cost = {}
    g_cost[start] = STEP_SIZE
    function = {}
    function[start] = calc_heuristic(start, goal) + g_cost[start]

    while len(open) > 0:
        # Get the node in open with the minimum fn score.
        curr = min_fn(open, function)
        blocked.append(curr)

        # If the goal has been found and optimised, print out solution
        if curr == goal:
            #print(g_cost.get(goal))
            return get_history(descended_from, curr, start)

        open.remove(curr)
        for neighbour in generate_branch(curr, size, blocked):
            curr_dist = g_cost[curr] + STEP_SIZE
            # If neighbour hasn't been seen yet, set it's g_cost to infinity
            if neighbour not in traversed:
                g_cost[neighbour] = inf
                traversed.append(neighbour)
                

            # If the distance calculated is lower than its previous cost...
            if curr_dist < g_cost[neighbour]:
                descended_from[neighbour] = curr
                g_cost[neighbour] = curr_dist
                function[neighbour] = curr_dist + calc_heuristic(neighbour, goal)

                if neighbour not in open:
                    open.add(neighbour)
    # If no solution is possible then return failure
    return 0

# Builds the path 
def get_history(descended_from, curr, start):
    soln = [curr]
    # Iterate through the descended dictionary
    while curr in list(descended_from.keys()):
        curr = descended_from[curr]
        soln.append(curr)
    return soln
    

# Calculates the start and goal by taking in the coordinates as lists 
# in [r, q] format.
def calc_heuristic(start, goal):
    r1 = start[0]
    q1 = start[1]
    r2 = goal[0]
    q2 = goal[1]

    # If heading top right or bottom left direction, use Manhattan distance
    if (q2 >= q1 and r2 >= r1 or q2 <= q1 and r2 <= r1):
        prediction = abs(q2 - q1) + abs(r2 - r1)
    # If goal is to the left of the start...
    else:
        prediction = max(abs(q2 - q1), abs(r2 - r1))

    return prediction


# Generate the possible branches from that node
def generate_branch(curr, size, blocked):
    branches = []
    max = size
    min = 0

    r1 = curr[0]
    q1 = curr[1]

    # Add possible branches into the branches list (Just brute force as there are only 6)
    if r1 - 1 >= min and (r1 - 1, q1) not in blocked:
        branches.append((r1 - 1, q1))

    if r1 + 1 < max  and (r1 + 1, q1) not in blocked:
        branches.append((r1 + 1, q1))
    
    if q1 - 1 >= min and (r1, q1 - 1) not in blocked:
        branches.append((r1, q1 - 1))
    
    if q1 + 1 < max and (r1, q1 + 1) not in blocked:
        branches.append((r1, q1 + 1))

    if q1 + 1 < max and r1 - 1 >= min and (r1 - 1, q1 + 1) not in blocked:
        branches.append((r1 - 1, q1 + 1))

    if q1 - 1 >= min and r1 + 1 < max and (r1 + 1, q1 - 1) not in blocked:
        branches.append((r1 + 1, q1 - 1))

    return branches

# Takes the data and transforms the board into a dictionary with keys
# as the tuple, and colour as the value. 
def board_dict(data):
    board = defaultdict()

    # Iterate through list and assign dictionary values
    for list in data.get("board"):
        tuple = (list[1], list[2])
        board[tuple] = list[0]
    return board

# Get node in openset with smallest fn value
def min_fn(openset, function):
    scores = []
    # Iteratte through the openset
    for hex in openset:
        scores.append((function[hex], hex))
    
    scores.sort(key=lambda y: y[0])

    return scores[0][1]
