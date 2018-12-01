import thresholding_labeling
from thresholding_labeling import CCL
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import math
from extract import pieceExtraction
from piece_class import Piece
from rotate_and_label_sides import label_sides
from rotate import rotate

#test = CCL(thresholding_labeling.thresholding_morph('all_pieces.jpg'))
#np.savetxt('ccl_out.csv', test[0], delimiter=',')

# ccl_out = np.loadtxt('ccl_out.csv', delimiter=',')
# img = cv2.imread('all_pieces.jpg')

# labels = np.unique(ccl_out)
# ccl_out[ccl_out == labels[1]] = 65
# pieceExtraction(ccl_out, img)

# for i in range(12):
# 	pre_original = np.load('new_piece_{}.npy'.format(i))
# 	plt.imshow(pre_original)
# 	plt.show()


all_pieces = []
for i in range(12):
	pre_original = np.load('new_piece_{}.npy'.format(i))
	original = np.zeros(pre_original.shape)
	original = np.array(original, dtype=np.uint8)
	for rowIdx, row in enumerate(pre_original):
	        for colIdx, pixel in enumerate(row):
	            if pixel != 0:
	                original[rowIdx][colIdx] = 255

	corners, edge_shape, side_lengths, img = label_sides(original, 'new_piece_{}'.format(i), 'new_piece_clr_{}.npy'.format(i))
	new_piece = Piece('new_piece_{}'.format(i), corners, edge_shape, side_lengths, img)
	all_pieces.append(new_piece)

'''
mid_pieces = []
#Isolates the middle pieces
for p in all_pieces:
	flat = False
	for edge in p.edge_shape:
		if(edge == 0):
			flat = True
	if(flat == False):
		mid_pieces.append(p.img)

plt.imshow(rotate(mid_pieces[0],270))
plt.show()
print(len(mid_pieces))

#Matching of the middle pieces '''
'''
p1 = mid_pieces[0]
p2 = mid_pieces[1]

scores = []

for x in range(4):
	if(p1.edges[x])


#Tip of the day: Dont't forget! Scream "I'm cumming" before executing your python file!'''

