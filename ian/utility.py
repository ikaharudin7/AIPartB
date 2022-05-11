# Evaluation function, takes the board at a given time before a move and evaluates how good that board is. 
from collections import defaultdict
from math import inf
import math

STEP_SIZE = 1

# Alpha Beta function for pruning: maximising_player = 1 or -1
def minimax(size, state, player_colour, maximising_player, alpha, beta, depth):
    if player_colour == "red":
        opponent = "blue"
    else:
        opponent = "red"

    # check depth
    if depth >= 2:
        eval = evaluation(player_colour, maximising_player, state, size)
        return (eval, state)
    
    if (maximising_player == 1):
        maxEval = -math.inf
        maxState = state # might be error here
        # go through each child of current state
        successors = successor(state, size, player_colour)
        for s in successors[0]: 
            eval = minimax(size, s, opponent, -1, alpha, beta, depth+1)[0]
            if eval > maxEval: # update the maximum state 
                maxState = s
            maxEval = max(maxEval, eval) 
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return (maxEval, maxState)

    else:
        minEval = math.inf
        minState = state # might be error here
        successors = successor(state, size, player_colour)
        for s in successors[0]: 
            eval = minimax(size, s, opponent, 1, alpha, beta, depth+1)[0]
            if eval < minEval:
                minState = s
            minEval = min(minEval, eval)
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
                # Add coord to current board
                current.append(coord)
                # Remove captured elements
                if len(disposable) > 0:
                    for item in disposable:
                        current.remove(item)
                
                states.append(current)
                new_coords.append((i, j, colour))

    return [states, new_coords]


# Return an evaluation of the given state of the board
def evaluation(player, maximising_player, state, size):
    if player == "red":
        opponent = "blue"
    else:
        opponent = "red"

    # Weighting for each step which a player is from winning
    #winning_weight = 6
    #curr_team = close_to_win(player, state, size)[2] * winning_weight
    #opp_team = close_to_win(opponent, state, size)[2] * winning_weight
    #step_difference = (curr_team - opp_team) * maximising_player

    # Get more detailed information about current state
    state_info = stateFeatures(state, size, player, opponent)

    # Weighting for the difference in number of pieces between player and opponent 
    one_piece_weight = 4
    player_total = state_info["player_total"]
    opponent_total = state_info["opponent_total"]
    piece_difference = (player_total - opponent_total) * one_piece_weight * maximising_player

    # Set weight = 8 for the number of player's pieces near the centre of the board
    centre_weight = 8
    centre_outside_diff = state_info["centred_players"] - state_info["outside_players"]
    centre_val = centre_outside_diff * centre_weight * maximising_player

    # Weighting for the number of opportunities to capture opponent 
    capture_weight = 32
    possible_captures = num_capture_positions(state, size, player, opponent)
    capture_val = 0
    if possible_captures > 0:
        capture_val = possible_captures * capture_weight * maximising_player

    print(piece_difference + centre_val + capture_val)
    return piece_difference + centre_val + capture_val


# Returns the number of coordinates on the board which player can capture
def num_capture_positions(board, size, player, opponent):
    num = 0
    for r in range(size):
        for q in range(size):
            if (r, q, player) not in board and (r, q, opponent) not in board:
                # check if this coordinate can capture
                if len(capture(board, (r, q, player))) > 0:
                    num += 1
    
    return num


# Returns information about a given board state, including total pieces and centred pieces
def stateFeatures(board, size, player, opponent):
    board_info = defaultdict()
    # these are the keys in this dictionary 
    num_player = 0
    num_opponent = 0
    maxPlayers_centre = 0
    outside_centre = 0

    if (size % 2 == 0): # board size is even
        # establish a centred region on the board, more than n hexes from the edge of board
        if (size == 4 or size == 6):
            n = 1
        elif (size == 8 or size == 10):
            n = 2
        elif (size == 12 or size == 14):
            n = 3

        for hex in board:
            # update the total number of player and opponent
            if hex[2] == player:
                num_player += 1
                # update the total number of maximising players around centre of the board
                if hex[0] < (size - n) and hex[0] >= n and hex[1] < (size - n) and hex[1] >= n:
                    maxPlayers_centre += 1
                else:
                    outside_centre += 1
            if hex[2] == opponent:
                num_opponent += 1
            

    else: 
        # establish the radius of the hexagon region based on the centre of the board
        if (size == 3 or size == 5):
            radius = 1
        elif (size == 7 or size == 9):
            radius = 2
        elif (size == 11 or size == 13):
            radius = 3
        elif (size == 15):
            radius = 4
        
        centre = size // 2 
        for hex in board:
            if hex[2] == player:
                num_player += 1
                if (centre + radius) >= hex[0] and (centre - radius) <= hex[0] and (centre
                 + radius) >= hex[1] and (centre - radius) <= hex[1]: 
                    maxPlayers_centre += 1
                else:
                    outside_centre += 1
            if hex [2] == opponent:
                num_opponent += 1



    board_info["player_total"] = num_player
    board_info["opponent_total"] = num_opponent
    board_info["centred_players"] = maxPlayers_centre
    board_info["outside_players"] = outside_centre

    return board_info


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
