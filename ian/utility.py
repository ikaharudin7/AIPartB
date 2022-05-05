# Evaluation function, takes the board at a given time before a move and evaluates how good that board is. 
from collections import defaultdict
from math import inf

STEP_SIZE = 1

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



# Detecting capture and returning captured elements - coord in (r, q, colour) format
def capture(self, board, coord):
    colour = coord[2]
    opp = "red"
    if coord[2] == "red":
        opp == "blue"
        
    captured = []

    # Check 6 possible options manually
    r = coord[0]
    q = coord[1]
    i = 0
    while i < 1:
        # Vertically above and go clockwise
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
        #colour = self.opp
        #opp = self.colour
        i += 1

    return captured







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
            print(g_cost.get(goal))
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