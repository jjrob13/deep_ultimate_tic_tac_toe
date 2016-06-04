import os, pickle
import numpy as np
import constants
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from utils import *


def move_grade_training():
	"""Train a model to predict if a given move is 'good' or not"""
	WHOSE_TURN_DIM = 1 #i.e. X or O
	BOARD_DIM = 81 #i.e. 9x9
	POSS_MOVE_DIM = 81 #ie. same as board size
	INPUT_DIM = WHOSE_TURN_DIM + BOARD_DIM + POSS_MOVE_DIM + POSS_MOVE_DIM #turn, board, last_move, new_move

	NB_EPOCH = 5

	#NOTE: X_PIECE always went first in the training data

	model = Sequential()
	model.add(Dense(2 * INPUT_DIM, input_dim=INPUT_DIM))
	model.add(Dense(INPUT_DIM))
	model.add(Dense(BOARD_DIM))
	model.add(Dense(1)) #predicting if the move was a 'winning' move or not
	model.add(Activation('softmax'))
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

	for batch in training_batch_generator():
		X = []
		y = []
		for game, winner in batch:
			#skipping ties for now, as we only want to track winning and losing moves
			if winner == constants.NO_PIECE:
				continue

			#all even values of i represent moves by X_PIECE

			#we are starting with an empty board, and hence, all zeros for previous move (i.e. there was no previous move)
			prev_move = np.zeros(81, dtype='int32')
			for i, board in enumerate(game[:-1]):
				#case 1, X won and this is an x move we are scoring
				if i % 2 == 0 and winner == constants.X_PIECE:
					y.append(1)
				#case 2, O won and this is an O move we are scoring
				elif (i % 2 == 1) and winner == constants.O_PIECE:
					y.append(1)
				else:
				#this is a loser's move
					y.append(0)

				turn = i % 2
				x1 = np.asarray(game[i].flatten()) #board
				x2 = prev_move
				row, col = get_move_for_boards(game[i], game[i + 1])
				move_idx = np.ravel_multi_index((row, col), (9, 9))
				x3 = to_categorical([move_idx], BOARD_DIM)[0]
				X.append(np.hstack([[turn], x1, x2, x3]))

				#we need to update the previous move for next iteration
				prev_move = x3


		X = np.asarray(X)
		y = np.asarray(y)
		model.fit(X, y, batch_size=32, nb_epoch=NB_EPOCH, validation_split=0.05)

	with open('keras_model.json', 'w') as f:
		f.write(model.to_json())


def move_predictor():
	"""
	build net to predict the best move to make
	NOTE: We are going to invert all O boards (if necessary), to have a standard board representation
		for the model
	"""
	BOARD_DIM = 81 #i.e. 9x9
	POSS_MOVE_DIM = 81 #ie. same as board size
	INPUT_DIM = BOARD_DIM + POSS_MOVE_DIM #board, last_move
	OUTPUT_DIM = POSS_MOVE_DIM #which move should we make?

	NB_EPOCH = 5

	#NOTE: X_PIECE always went first in the training data
	model = Sequential()
	model.add(Dense(2 * INPUT_DIM, input_dim=INPUT_DIM))
	model.add(Dropout(0.2))
	model.add(Dense(INPUT_DIM))
	model.add(Dropout(0.2))
	model.add(Dense(INPUT_DIM))
	model.add(Dropout(0.2))
	model.add(Dense(OUTPUT_DIM))
	model.add(Activation('softmax'))
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	for batch in training_batch_generator():
		X = []
		y = []
		for game, winner in batch:
			#skipping ties for now, as we only want to track winning and losing moves
			if winner == constants.NO_PIECE:
				continue
			#we are only trying to predict "winning" moves, so we will only be looking at moves that the
			#winner made
			if winner == constants.X_PIECE:
				i = 0
			else:
				i = 1
				#let's invert all of the boards if O_PIECE won, so we have a standardized data representation
				game = invert_game(game)

			#all even values of i represent moves by X_PIECE
			while i < len(game) - 1:
				if i == 0:
					prev_move = np.zeros(81, dtype='int32')
				else:
					row, col = get_move_for_boards(game[i - 1], game[i])
					prev_move_idx = np.ravel_multi_index((row, col), (9, 9))
					prev_move = to_categorical([prev_move_idx], BOARD_DIM)[0]
					

				x1 = np.asarray(game[i].flatten()) #board
				x2 = prev_move
				X.append(np.hstack([x1, x2]))

				#this is our label, i.e. which move should we make?
				row, col = get_move_for_boards(game[i], game[i + 1])
				move_idx = np.ravel_multi_index((row, col), (9, 9))

				y.append([move_idx])
				i += 2


		X = np.asarray(X)
		y = np.asarray(y)
		model.fit(X, y, batch_size=32, nb_epoch=NB_EPOCH, validation_split=0.05)

	with open('keras_model.json', 'w') as f:
		f.write(model.to_json())


if __name__ == '__main__':
	move_predictor()
