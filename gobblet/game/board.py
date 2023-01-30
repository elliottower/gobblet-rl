import numpy as np

class Board:
    def __init__(self):
        # internally self.board.squares holds a representation of the gobblet board, consisting of three stacked 3x3 boards, one for each piece size.
        # We flatten it for simplicity, from [3,3,3] to [27,]

        # An empty board with the first small piece in the top left corner and the second large piece in the bottom right corner would be:
        # [[1, 0, 0, 0, 0, 0, 0, 0, 0],
        #  [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #  [0, 0, 0, 0, 0, 0, 0, 0, 6]]
        # In each of the three levels (small, medum, and large pieces), an empty 3x3 board is [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # where indexes are column wise order
        # 0 3 6
        # 1 4 7
        # 2 5 8

        # Reshaped to three dimensions, this would look like:
        # [[[1, 0, 0],
        #   [0, 0, 0],
        #   [0, 0, 0]],
        #  [[0, 0, 0],
        #   [0, 0, 0],
        #   [0, 0, 0]],
        #  [[0, 0, 0],
        #   [0, 0, 0],
        #   [0, 0, 6]



        # empty -- 0
        # player 0 -- 1
        # player 1 -- -1 # Default: 2
        self.squares = np.zeros(27)
        self.squares_preview = np.zeros(27)

        # precommute possible winning combinations
        self.calculate_winners()

    def setup(self):
        self.calculate_winners()

    def get_action_from_pos_piece(self, pos, piece):
        if pos in range(9) and piece in range(1,7):
            return 9 * (piece - 1) + pos
        else:
            return -1

    # Return an action if an agent can place of the specified size in the specified location
    # Checks both possible pieces of that size, and returns -1 if neither can be placed (i.e., they are both covered)
    def get_action(self, pos, piece_size, agent_index):
        piece1 = piece_size * 2 - 1
        piece2 = piece_size * 2
        action1 = pos + 9 * (piece1 - 1)
        action2 = pos + 9 * (piece2 - 1)
        if self.is_legal(action1, agent_index):
            return action1
        elif self.is_legal(action2, agent_index):
            return action2
        else:
            return -1

    # To determine the position from an action, we take the number modulo 9, resulting in a number 0-8
    def get_pos_from_action(self, action):
        return action % 9

    # To determine the piece from an action i, we use floor division by 9 (i // 9), resulting in a number 0-5, where 1-2 represent small pieces, 3-4 represent medium pieces, and 5-6 represent large pieces.
    def get_piece_from_action(self, action):
        return (action // 9) + 1

    # To determine the size of a given piece p (1-6), we use floor division by 2 (p + 1 // 2) resulting in a number 1-3
    def get_piece_size_from_action(self, action):
        piece = self.get_piece_from_action(action)
        return (piece + 1) // 2

    # Returns the index on the board [0-26] for a given action
    def get_index_from_action(self, action):
        pos = self.get_pos_from_action(action)  # [0-8]
        piece_size = self.get_piece_size_from_action(action) # [1-3]
        return pos + 9 * (piece_size - 1) # [0-26]


    # Return true if an action is legal, false otherwise.
    def is_legal(self, action, agent_index=0):
        pos = self.get_pos_from_action(action) # [0-8]
        piece = self.get_piece_from_action(action) # [1-6]
        piece_size = self.get_piece_size_from_action(action) # [1-3]
        agent_multiplier = 1 if agent_index == 0 else -1

        board = self.squares.reshape(3, 9)
        # Check if this piece has been placed (if the piece number occurs anywhere on the level of that piece size)
        if any(board[piece_size-1] == piece * agent_multiplier):
            current_loc = np.where(board[piece_size-1] == piece * agent_multiplier)[0] # Returns array of values where piece is placed
            if len(current_loc) > 1:
                raise Exception("Error: piece has been used twice")
            else:
                current_loc = current_loc[0] # Current location [0-27]
            # If this piece is currently covered, moving it is not a legal action
            if self.check_covered()[current_loc] == 1:
                return False

        # If this piece has not been placed
        # Check if the spot on the flat 3x3 board is open (we can definitely place in that case)
        flatboard = self.get_flatboard()
        if flatboard[pos] == 0:
            return True
        else:
            existing_piece_number = flatboard[pos] # [1-6]
            existing_piece_size = (abs(existing_piece_number) + 1) // 2 # [1-3]
            if piece_size > existing_piece_size:
                return True
            else:
                return False

    # Update the board with an agent's move
    def play_turn(self, agent_index, action):
        piece = self.get_piece_from_action(action)
        if agent_index == 1:
            piece = piece * -1
        index = self.get_index_from_action(action)

        # First: check if a move is legal or not
        if not self.is_legal(action, agent_index):
            print("ILLEGAL MOVE: ", action)
            return
        # If piece has already been placed, clear previous location
        if piece in self.squares:
            old_index = np.where(self.squares == piece)[0][0]
            self.squares[old_index] = 0
        self.squares[index] = piece
        return

    # Expects flat [9,] length array, from tic-tac-toe code
    def calculate_winners(self):
        winning_combinations = []
        indices = [x for x in range(0, 9)]

        # Vertical combinations
        winning_combinations += [
            tuple(indices[i : (i + 3)]) for i in range(0, len(indices), 3)
        ]

        # Horizontal combinations
        winning_combinations += [
            tuple(indices[x] for x in range(y, len(indices), 3)) for y in range(0, 3)
        ]

        # Diagonal combinations
        winning_combinations.append(tuple(x for x in range(0, len(indices), 4)))
        winning_combinations.append(tuple(x for x in range(2, len(indices) - 1, 2)))

        self.winning_combinations = winning_combinations

    # returns flattened board consisting of only top pieces (excluding pieces which are gobbled by other pieces)
    def get_flatboard(self):
        flatboard = np.zeros(9)
        board = self.squares.reshape(3, 9)
        for i in range(9):  # For every square in the 3x3 grid, find the topmost element (largest piece)
            top_piece_size = (np.amax(
                abs(board[:, i])))  # [-3, 2, 0] denotes a large piece gobbling a medium piece, this will return 3
            top_piece_index = list(abs(board[:, i])).index(
                top_piece_size)  # Get the row of the top piece (have to index [0]
            top_piece_color = np.sign(board[
                                          top_piece_index, i])  # Get the color of the top piece: 1 for player_1 and -1 for player_2, 0 for neither
            flatboard[i] = top_piece_color * top_piece_size # Simplify the board into only the top elements
        return flatboard

    # returns:
    # 0 for no winner
    # 1 -- agent 1 wins
    # -1 -- agent 2 wins
    def check_for_winner(self):
        winner = 0
        flatboard = self.get_flatboard()
        for combination in self.winning_combinations:
            states = []
            for index in combination:
                states.append(flatboard[index]) # default: self.squares[index]
            if all(x > 0 for x in states):
                winner = 1
            if all(x < 0 for x in states): # Change to -1 probably?
                winner = -1
        return winner

    def check_game_over(self):
        winner = self.check_for_winner()
        if winner in [1, -1]:
            return True
        else:
            return False

    def check_covered(self): # Return a 27 length array indicating which positions have a piece which is covered
        board = self.squares.reshape(3, 9)
        covered = np.zeros((3, 9))
        for i in range(9): # Check small pieces
            if board[0, i] != 0 and (board[1, i] != 0 or board[2, i] != 0): # If there is a small piece, and either a large or medium piece covering it
                covered[0, i] = 1
        for i in range(9): # Check medium pieces
            if board[1, i] != 0 and board[2, i] != 0: # If there is a meidum piece and a large piece covering it
                covered[1, i] = 1
        covered[2, :] = 0 # Large pieces can't be covered
        # Doesn't matter about what color is covering it, you can't move that piece this turn (allows self-gobbling)
        return covered.flatten()

# DEBUG
    def print_pieces(self):
        open_indices = [i for i in range(len(self.squares)) if self.squares[i] == 0]
        open_squares = [np.where(self.get_flatboard() == 0)[0]]
        occupied_squares = [i % 9 for i in range(len(self.squares)) if self.squares[i] != 0] # List with entries 0-9
        movable_squares = [i % 9 for i in occupied_squares if self.check_covered()[i] == 0] # List with entries 0-9
        covered_squares = [i % 9 for i in np.where(self.check_covered() == 1)[0] ] # List with entries 0-9
        print("open_indices: ", open_indices)
        print("open_squares: ", open_squares)
        print("squares with pieces: ", occupied_squares)
        print("squares with uncovered pieces: ", movable_squares)
        print("squares with covered pieces: ", covered_squares)

    def __str__(self):
        return str(self.squares)
