from Agent import Agent
from GameBoard import GameBoard
import numpy as np

EMPTY_WEIGHT = 500000
SMOOTHNESS_WEIGHT = 3

class ExpectimaxAgent(Agent):
    def __init__(self, max_depth=4, heuristic="sum"):
        self.heuristic_algorithm = heuristic
        self.max_depth = max_depth
        pass

    def play(self, board:GameBoard):
        best_move, _ = self.maximize_utility(board)
        return best_move

    def maximize_utility(self, board, depth=0):
        moves = board.get_available_moves()
        moves_boards = []

        for m in moves:
            m_board = board.clone()
            m_board.move(m)
            moves_boards.append((m, m_board))

        max_utility = float('-inf')
        best_direction = None

        for mb in moves_boards:
            utility = self.utility_after_oponent_move(mb[1], depth + 1)

            if utility >= max_utility:
                max_utility = utility
                best_direction = mb[0]

        return best_direction, max_utility

    def utility_after_oponent_move(self, board, depth)->int:
        empty_cells = board.get_available_cells()
        n_empty = len(empty_cells)

        if depth >= self.max_depth:
            return self.heuristic_utility(board)

        if n_empty >= 6 and depth >= 3:
            return self.heuristic_utility(board)
            
        if n_empty == 0:
            _, utility = self.maximize(board, depth + 1)
            return utility 

        possible_tiles = []

        chance_2 = (.9 * (1 / n_empty))
        chance_4 = (.1 * (1 / n_empty))
        
        for empty_cell in empty_cells:
            possible_tiles.append((empty_cell, 2, chance_2))
            possible_tiles.append((empty_cell, 4, chance_4))

        avg_utility = 0

        for t in possible_tiles:
            t_board = board.clone()
            t_board.insert_tile(t[0], t[1])
            _, utility = self.maximize_utility(t_board, depth + 1)

            avg_utility += utility

        avg_utility /= len(possible_tiles)

        return avg_utility

    def heuristic_utility(self, board: GameBoard)->int:
       
        if (self.heuristic_algorithm == "smoothness"):
            return self.get_smoothness(board)
        elif (self.heuristic_algorithm == "value"):
            return self.get_board_value(board)
        elif (self.heuristic_algorithm == "empty"):
            return self.get_empty_value(board)
        
        return self.get_smoothness(board) + self.get_board_value(board) + self.get_empty_value(board)

    def get_smoothness(self, board:GameBoard)->int:
        s_grid = np.sqrt(grid)
        smoothness = 0
        smoothness += np.sum(np.abs(s_grid[::,0] - s_grid[::,1]))
        smoothness += np.sum(np.abs(s_grid[::,1] - s_grid[::,2]))
        smoothness += np.sum(np.abs(s_grid[::,2] - s_grid[::,3]))
        smoothness += np.sum(np.abs(s_grid[0,::] - s_grid[1,::]))
        smoothness += np.sum(np.abs(s_grid[1,::] - s_grid[2,::]))
        smoothness += np.sum(np.abs(s_grid[2,::] - s_grid[3,::]))
        grid = board.grid
        return - smoothness ** SMOOTHNESS_WEIGHT
    
    def get_board_value(self, board:GameBoard)->int:
        return np.sum(np.power(board.grid, 2))

    def get_empty_value(self, board:GameBoard)->int:
        empty_cells = board.get_available_cells()
        n_empty = len(empty_cells)

        return EMPTY_WEIGHT * n_empty
