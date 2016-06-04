from collections import defaultdict
import constants, copy
import numpy as np

class Board(object):
	def __init__(self):
		self.board = np.full((9, 9), constants.NO_PIECE, dtype='int32')

		#so we do not have to compute this every time
		#maps from (i, j) to available moves in tic-tac-toe cell i, j; i == row_num, j == col_num
		self.available_moves = defaultdict(set)
		self.next_moves = set() #set of all moves that can be selected
		for i in range(len(self.board)):
			for j in range(len(self.board[0])):
				row, col = Board.subboard_for_coord(i, j)
				self.available_moves[row, col].add((i, j))
				self.next_moves.add((i, j))

	@staticmethod
	def subboard_for_coord(row, col):
		board_row = row / 3
		board_col = col / 3
		return board_row, board_col

	def available_moves_in_cell(self, cell_row, cell_col):
		"""
		Returns the moves that can be made in this specific cell.
		Returns the empty set if the cell has already been won, or there is no additional moves to be made.
		"""

		if self.subcell_winner(cell_row, cell_col) != constants.NO_PIECE:
			return set()

		start_row = cell_row * 3
		start_col = cell_col * 3
		#check if there are no additional moves
		if not constants.NO_PIECE in self.board[start_row:start_row + 3, start_col:start_col + 3]:
			return set()

		return self.available_moves[cell_row, cell_col]

	def update_available_moves_for_piece(self, piece_row, piece_col):
		"""NOTE: Must be called after actual piece has been placed on board"""
		cell_row, cell_col = Board.subboard_for_coord(piece_row, piece_col)

		if self.subcell_winner(cell_row, cell_col) != constants.NO_PIECE:
			self.available_moves[cell_row, cell_col].clear()
			return

		start_row = cell_row * 3
		start_col = cell_col * 3
		#check if there are no additional moves
		if not constants.NO_PIECE in self.board[start_row:start_row + 3, start_col:start_col + 3]:
			self.available_moves[cell_row, cell_col].clear()
			return

		#just remove the piece from the map
		self.available_moves[cell_row, cell_col].remove((piece_row, piece_col))

	
	def move_made(self, row, col):
		"""Updates the set of possible moves that can be made"""
		self.update_available_moves_for_piece(row, col)

		#we need to compute what the proper set of next moves will be
		next_cellcol = col % 3
		next_cellrow = row % 3

		#case 1, there are still moves to be made in the cell
		self.next_moves = copy.copy(self.available_moves[next_cellrow, next_cellcol])

		#case 2, that cell is won/tied, so the player can move anywhere
		if not self.next_moves:
			for v in self.available_moves.values():
				self.next_moves |= v

		
	def add_piece(self, row, col, piece):
		assert piece == constants.X_PIECE or piece == constants.O_PIECE 
		if not (0 <= row < len(self.board)):
			raise ValueError('Invalid row of {}'.format(row))
		if not (0 <= col < len(self.board[0])):
			raise ValueError('Invalid col of {}'.format(col))

		if self.board[row, col] != constants.NO_PIECE:
			raise ValueError('Piece already at (%d, %d)' % (row, col))

		if not (row, col) in self.next_moves:
			raise ValueError('Not allowed to move to position (%d, %d)' % (row, col))

		self.board[row, col] = piece
		self.move_made(row, col)
		
		return True

	def game_over(self):
		"""Helper function to return bool to indicate completion of game"""
		return all(not x for x in self.available_moves.values()) or (self.board_winner() != constants.NO_PIECE)

	def board_winner(self):
		"""Returns the winner of the entire board, if one exists"""
		collapsed_board = np.full((3, 3), constants.NO_PIECE, dtype='int32')
		for i in range(3):
			for j in range(3):
				collapsed_board[i, j] = self.subcell_winner(i, j)

		return Board.simple_tic_tac_toe_winner(collapsed_board)
	def subcell_winner(self, cell_row_num, cell_col_num):
		"""
		Returns the winner of the cell if one of the sub tic-tac-toe games has been won and NO_PIECE (i.e. -1)
		otherwise
		0 <= row_num <= 2

		"""
		assert 0 <= cell_row_num <= 2 and 0 <= cell_col_num <= 2

		start_row = cell_row_num * 3
		start_col = cell_col_num * 3
		return Board.simple_tic_tac_toe_winner(self.board[start_row:start_row+3, start_col:start_col+3])

	@staticmethod
	def simple_tic_tac_toe_winner(simple_board):
		"""Method to determine who, if anyone, won the sub-tic-tac-toe game"""
		assert simple_board.shape == (3, 3)

		#case 1, won column
		for col_num in range(simple_board.shape[1]):
			#check if someone won this column
			if simple_board[0, col_num] != constants.NO_PIECE and\
				all(x == simple_board[0, col_num] for x in simple_board[:, col_num]):
				return simple_board[0, col_num]

		#case 2, row victory
		for row_num in range(simple_board.shape[0]):
			if simple_board[row_num, 0] != constants.NO_PIECE and\
				all(x == simple_board[row_num, 0] for x in simple_board[row_num, :]):
				return simple_board[row_num, 0]

		#case 3, diagonal victory
		if simple_board[0, 0] != constants.NO_PIECE and\
			(simple_board[0, 0] == simple_board[1, 1] == simple_board[2, 2]):
			return simple_board[0, 0]

		if simple_board[0, 2] != constants.NO_PIECE and\
			(simple_board[0, 2] == simple_board[1, 1] == simple_board[2, 0]):
			return simple_board[0, 2]


		#case 4, no winner
		return constants.NO_PIECE

	def __str__(self):
		return str(self.board)

def simple_winner_test():
	#column test
	for i in range(3):
		test_board = np.full((3, 3), constants.NO_PIECE, dtype='int32')
		test_board[:, i] = 3 * [constants.X_PIECE]
		assert Board.simple_tic_tac_toe_winner(test_board) == constants.X_PIECE

	#row test
	for i in range(3):
		test_board = np.full((3, 3), constants.NO_PIECE, dtype='int32')
		test_board[i, :] = 3 * [constants.O_PIECE]
		assert Board.simple_tic_tac_toe_winner(test_board) == constants.O_PIECE

	#diagonal test
	test_board = np.full((3, 3), constants.NO_PIECE, dtype='int32')
	assert Board.simple_tic_tac_toe_winner(test_board) == constants.NO_PIECE
	test_board[0, 0] = constants.X_PIECE
	test_board[1, 1] = constants.X_PIECE
	test_board[2, 2] = constants.X_PIECE
	assert Board.simple_tic_tac_toe_winner(test_board) == constants.X_PIECE

	test_board[2, 0] = constants.O_PIECE
	test_board[0, 2] = constants.O_PIECE
	assert Board.simple_tic_tac_toe_winner(test_board) == constants.X_PIECE
	
	test_board[1, 1] = constants.O_PIECE
	assert Board.simple_tic_tac_toe_winner(test_board) == constants.O_PIECE

if __name__ == '__main__':
	from pprint import pprint
	b = Board()
	print b.available_moves.keys()
