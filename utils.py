
import os, pickle
import numpy as np
import constants
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

def transpose_batch(batch):
	transposed_batch = []
	for boards, winner in batch:
		transposed_boards = [np.transpose(b) for b in boards]
		transposed_batch.append((transposed_boards, winner))
	return transposed_batch

def training_batch_generator():
	TRAINING_DIR = 'training_data'
	for filename in os.listdir(TRAINING_DIR):
		print 'loading %s' % filename
		path = os.path.join(TRAINING_DIR, filename)
		with open(path, 'rb') as f:
			data = pickle.load(f)
		yield data
		
		print 'transposing %s' % filename
		transposed_data = transpose_batch(data)
		del data #free up some mem
		yield transposed_data

def get_move_for_boards(board1, board2):
	for i in range(len(board1)):
		for j in range(len(board1[0])):
			if board1[i][j] != board2[i][j]:
				return i, j
				
	assert ValueError('No move was detected!')

def invert_board(board):
	"""Change all O's to X's and vice versa"""
	result = np.full(board.shape, constants.NO_PIECE, dtype='int32')
	for i in range(len(board)):
		for j in range(len(board[0])):
			if board[i][j] == constants.X_PIECE:
				result[i][j] = constants.O_PIECE
			elif board[i][j] == constants.O_PIECE:
				result[i][j] = constants.X_PIECE
	return result
				
def invert_game(game):
	"""Change the piece labels for a given game.
	That is, all O's are changed to X's and vice versa
	"""
	return [invert_board(b) for b in game]
