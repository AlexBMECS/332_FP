import numpy as np
from rotate_and_label_sides import label_sides

class Piece:
	def __init__(self, name, corners, edge_shape, side_lengths, img):

		self.name = name
		self.corners = corners
		self.edge_shape = edge_shape
		self.side_lengths = side_lengths
		self.img = img

all_pieces = []
for i in range(12):
	pre_original = np.loadtxt('piece_{}'.format(i), delimiter=',')
	original = np.zeros(pre_original.shape)
	original = np.array(original, dtype=np.uint8)
	for rowIdx, row in enumerate(pre_original):
	        for colIdx, pixel in enumerate(row):
	            if pixel != 0:
	                original[rowIdx][colIdx] = 255

	corners, edge_shape, side_lengths, img = label_sides(original, 'piece_{}'.format(i), 'piece_clr_{}.npy'.format(i))
	new_piece = Piece('piece_{}'.format(i), corners, edge_shape, side_lengths, img)
	all_pieces.append(new_piece)