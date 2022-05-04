
from cmath import inf
from collections import defaultdict
from hmac import digest
from pickle import FALSE, TRUE
from tkinter import N
from numpy import ceil
from referee.board import Board

STEP_SIZE = 1

class Player:
    def __init__(self, player, n):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "red" if your player will
        play as Red, or the string "blue" if your player will play
        as Blue.
        """
        # put your code here
        self.colour = player
        self.moves = 0
        self.board = []
        self.n = n
        self.game = Board
        

        # Set the directions of the player.
        if player == "red":
            self.opp = "blue"
            self.vertical = True
        elif player == "blue": 
            self.opp = "red"
            self.vertical = False

        
        


    def action(self):
        """
        Called at the beginning of your turn. Based on the current state
        of the game, select an action to play.
        """
        # put your code here
        self.moves += 1

        # First move option
        if self.moves == 1 and self.colour == "red":
            if self.n % 2 == 0:
                return ("PLACE", int(ceil(self.n / 2)), int(ceil(self.n / 2)))
            else: 
                return ("PLACE", int(ceil(self.n / 2 - 1)), int(ceil(self.n / 2)))
        # Second move option
        elif self.moves == 1 and self.colour == "blue":
            return ("STEAL", )
        
        # Use A-Star initially.
        if self.vertical:
            choose = aStar((self.n - 1, 1), (0, 2), self.n, self.board)

        else:
            choose = aStar((1, self.n - 1), (2, 0), self.n, self.board)
        

        for tup in choose:
            # Problem here as self.board has different format. 
            if (tup[0], tup[1], "red") not in self.board and (tup[0], tup[1], "blue") not in self.board:
                cs = tup
                break

        return ("PLACE", cs[0], cs[1])



    # Evaluation function
    def eval(self, board, ):
        # OFFENCE - add values which fill up along the AStar path
        # (Give more value to a state which has 2 more hexes to go compared to 4 more to win)
        

        # DEFENCE - add greatest value to a block/capture if opp is 1 or 2 away from winning

        # Find a different AStar Path/ Block
         
         
        return



    def turn(self, player, action):
        """
        Called at the end of each player's turn to inform this player of 
        their chosen action. Update your internal representation of the 
        game state based on this. The parameter action is the chosen 
        action itself. 
        
        Note: At the end of your player's turn, the action parameter is
        the same as what your player returned from the action method
        above. However, the referee has validated it at this point.
        """
        # put your code 
        
        

        # Updates which elements are in the board at the moment. 
        if (len(action) > 2):
            # Use apply_captures to update the board
            remove = capture(self, self.board, (action[1], action[2], player))
            for item in remove:
                self.board.remove(item)

            self.board.append((action[1], action[2], player))
        else:
            temp = self.board[0] 
            self.board = [(temp[0], temp[1], player)]

        print(self.board)

        # Keep track of board in array form like part A, with occupied tuples there, with the colour. 
        

        return action

    


# Detecting capture and returning captured elements - coord in (r, q, colour) format
def capture(self, board, coord):
    colour = self.colour
    opp = self.opp
    captured = []

    # Check 6 possible options manually
    r = coord[0]
    q = coord[1]

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
    
    return captured







# A star algorithm
def aStar(start, goal, size, board):
    # Keep track of which hexes have been blocked
    blocked = []
    for hex in board:
        blocked.append(hex)

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