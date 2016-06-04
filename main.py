from board import Board
import constants, random
from tqdm import tqdm
import copy, pickle, time
from utils import invert_game, invert_board, get_move_for_boards, transpose_batch
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import numpy as np

def my_normalize(a):
	sum_a = sum(a)
	x = [p/float(sum_a) for p in a]
	return x

def user_play():
	board = Board()
	turn = constants.X_PIECE
	print board
	while not board.game_over():
		row, col = [int(x) for x in raw_input().split()]
		board.add_piece(row, col, turn)

		turn = (turn + 1) % 2
		print board

def random_play():
	random.seed()
	num_games = 20000
	batch_size = 10000
	games = [] #game = (list of board configs, winner)

	current_batch = 0
	for i in tqdm(range(num_games)):
		board = Board()
		boards = [copy.copy(board.board)]

		turn = constants.X_PIECE
		while not board.game_over():
			row, col = random.sample(board.next_moves, 1)[0]
			board.add_piece(row, col, turn)

			turn = (turn + 1) % 2
			boards.append(copy.copy(board.board))

		games.append((boards, board.board_winner()))
		current_batch += 1
		
		if current_batch == batch_size:
			with open('{}-saved_games.pkl'.format(time.time()), 'wb') as f:
				pickle.dump(games, f)
			current_batch = 0
			games = []

def generate_random_games(num_games, seed=1337):
	random.seed(seed)
	games = []
	for i in tqdm(range(num_games)):
		board = Board()
		boards = [copy.copy(board.board)]

		turn = constants.X_PIECE
		while not board.game_over():
			row, col = random.sample(board.next_moves, 1)[0]
			board.add_piece(row, col, turn)

			turn = (turn + 1) % 2
			boards.append(copy.copy(board.board))

		games.append((boards, board.board_winner()))

	return games
	
def train_model_on_games(model, games, nb_epoch=5):
	"""Fit the model on the given games"""

	#remove ties
	games = [(g, w) for g, w in games if w != constants.NO_PIECE]
	X = []
	y = []
	for game, winner in games:
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
			#there was no previous move at move #1
			if i == 0:
				prev_move = np.zeros(81, dtype='int32')
			else:
				row, col = get_move_for_boards(game[i - 1], game[i])
				prev_move_idx = np.ravel_multi_index((row, col), (9, 9))
				prev_move = to_categorical([prev_move_idx], 81)[0]
				

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
	model.fit(X, y, batch_size=32, nb_epoch=nb_epoch, validation_split=0.05)


def trained_model_play():
	"""
	NOTE: The models expect the board to be presented as player X's turn
	
	
	Algo:
	1. Start with 20000 randomly generated games
	2. Train a model to predict "winning" moves
	3. Generate 20000 new games, playing the model against itself
	4. Go to 2
	"""

	BOARD_DIM = 81 #i.e. 9x9
	POSS_MOVE_DIM = 81 #ie. same as board size
	INPUT_DIM = BOARD_DIM + POSS_MOVE_DIM #board, last_move
	OUTPUT_DIM = POSS_MOVE_DIM #which move should we make?

	NB_EPOCH = 5
	NB_ITER = 5 #number of reinforcement learning iterations

	#NOTE: X_PIECE always went first in the training data
	model = Sequential()
	model.add(Dense(2 * INPUT_DIM, input_dim=INPUT_DIM, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(2 * INPUT_DIM, activation='tanh'))
	model.add(Dropout(0.2))
	model.add(Dense(OUTPUT_DIM))
	model.add(Activation('softmax'))
	model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])


	num_games = 20000
	#game = (list of board configs, winner)
	games = generate_random_games(num_games)

	#we only want games with a definitive winner
	won_games = [(g, w) for g, w in games if w != constants.NO_PIECE]
	print 'Using {} games that have winner'.format(len(won_games))

	#we can easily scale up the number of games by transposing them
	won_games.extend(transpose_batch(won_games))

	train_model_on_games(model, won_games, nb_epoch=NB_EPOCH)

	for j in range(NB_ITER):
		games = []
		for i in range(num_games):
			board = Board()
			boards = [board.board]

			prev_move = np.zeros(BOARD_DIM, dtype='int32')
			turn = constants.X_PIECE
			while not board.game_over():
				if turn == constants.X_PIECE:
					x1 = np.asarray(board.board.flatten())
				elif turn == constants.O_PIECE:
					board_rep = invert_board(board.board)
					x1 = np.asarray(board_rep.flatten())
				else:
					raise ValueError('Mistakes have been made')

				x2 = prev_move
				X = np.asarray([np.hstack([x1, x2])])
				probs =  model.predict_proba(X)[0]
				#we need to eliminate any moves that are not allowed
				probs = [p if np.unravel_index(p_i, (9, 9)) in board.next_moves else 0\
					for p_i, p in enumerate(probs)]

				probs = my_normalize(probs)

				idx = range(len(probs))
				#predicted move to make
				move_idx = np.random.choice(idx, p=probs)
				
				row, col = np.unravel_index(move_idx, (9, 9))
				board.add_piece(row, col, turn)
				turn = (turn + 1) % 2
				prev_move = to_categorical([move_idx], 81)[0]

				boards.append(copy.copy(board.board))

			games.append((boards, board.board_winner()))

		won_games = [(g, w) for g, w in games if w != constants.NO_PIECE]
		print 'Using {} games that have winner after reinforcement iter {}'.format(len(won_games), j)
		#we can easily scale up the number of games by transposing them
		won_games.extend(transpose_batch(won_games))
		train_model_on_games(model, won_games, nb_epoch=NB_EPOCH)
	
	with open('keras_model.json', 'w') as f:
		f.write(model.to_json())
	
	model.save_weights('model_weights.h5')

def play_against_model():
	from keras.models import model_from_json
	model = model_from_json(open('keras_model.json').read())
	model.load_weights('model_weights.h5')
	model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])
	board = Board()

	prev_move = np.zeros(81, dtype='int32')
	turn = constants.X_PIECE
	while not board.game_over():
		if turn == constants.X_PIECE:
			x1 = np.asarray(board.board.flatten())
			x2 = prev_move
			X = np.asarray([np.hstack([x1, x2])])
			probs =  model.predict_proba(X)[0]
			#we need to eliminate any moves that are not allowed
			probs = [p if np.unravel_index(p_i, (9, 9)) in board.next_moves else 0\
				for p_i, p in enumerate(probs)]

			probs = my_normalize(probs)

			idx = range(len(probs))
			#predicted move to make
			move_idx = np.random.choice(idx, p=probs)
			
			row, col = np.unravel_index(move_idx, (9, 9))
			board.add_piece(row, col, turn)
		elif turn == constants.O_PIECE:
			print 'Allowed:'
			print board.next_moves
			try:
				row, col = [int(x) for x in raw_input('User move:').split()]
				board.add_piece(row, col, turn)
			except ValueError:
				print 'Try again'
				continue

		else:
			raise ValueError('Mistakes have been made')

		turn = (turn + 1) % 2
		print board

	print '{} Won the game!'.format(board.board_winner())


def random_against_model(ngames=100):
	from keras.models import model_from_json
	random.seed()
	model = model_from_json(open('keras_model.json').read())
	model.load_weights('model_weights.h5')
	model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])
	outcomes = []
	for _ in xrange(ngames):
		board = Board()

		prev_move = np.zeros(81, dtype='int32')
		turn = constants.X_PIECE
		while not board.game_over():
			if turn == constants.X_PIECE:
				x1 = np.asarray(board.board.flatten())
				x2 = prev_move
				X = np.asarray([np.hstack([x1, x2])])
				probs =  model.predict_proba(X)[0]
				#we need to eliminate any moves that are not allowed
				probs = [p if np.unravel_index(p_i, (9, 9)) in board.next_moves else 0\
					for p_i, p in enumerate(probs)]

				probs = my_normalize(probs)

				idx = range(len(probs))
				#predicted move to make
				move_idx = np.random.choice(idx, p=probs)
				
				row, col = np.unravel_index(move_idx, (9, 9))
				board.add_piece(row, col, turn)
			elif turn == constants.O_PIECE:
				try:
					row, col = random.sample(board.next_moves, 1)[0]
					board.add_piece(row, col, turn)
				except ValueError:
					print 'Try again'
					continue

			else:
				raise ValueError('Mistakes have been made')

			turn = (turn + 1) % 2

		print '{} Won the game!'.format(board.board_winner())
		outcomes.append(board.board_winner())

	print 'AI Won {:0.02f}% of the games!'.format(sum(1 if i == constants.X_PIECE else 0 for i in outcomes)/float(len(outcomes)))
	print '{:0.02f}% ties'.format(sum(1 if i == -1 else 0 for i in outcomes)/float(len(outcomes)))
	

if __name__ == '__main__':
	random_against_model()
