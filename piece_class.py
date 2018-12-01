import numpy as np
from rotate_and_label_sides import label_sides

class Piece:
	def __init__(self, name, corners, edge_shape, side_lengths, img):

		#Piece ID: e.g. new_piece_0
		self.name = name
		#Corner starts with bottom right corner
		self.corners = corners
		#Edge shapes start with bottom edge
		self.edge_shape = edge_shape
		#Side lengths start with bottom edge
		self.side_lengths = side_lengths
		#Img is the rotated, colored image
		self.img = img