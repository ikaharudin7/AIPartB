
from cmath import inf
from collections import defaultdict
from numpy import ceil
from referee.board import Board
import ian.utility as ian

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
        steps = ian.close_to_win(self, self.colour)
        print("min steps is", steps)

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
        
        return ian.evalfunc(self.n, self.board, self.vertical, self.colour)
         

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

        print(ian.evaluation(self))

        # Updates which elements are in the board at the moment. 
        if (len(action) > 2):
            # Use apply_captures to update the board
            remove = ian.capture(self, self.board, (action[1], action[2], player))
            if len(remove) > 1:
                
                for item in remove:
                    self.board.remove(item)
                

            self.board.append((action[1], action[2], player))

        # If steal has occurred. 
        else:
            temp = self.board[0] 
            self.board = [(temp[1], temp[0], player)]

        print(self.board)
        # Keep track of board in array form like part A, with occupied tuples there, with the colour. 

        return action

    

