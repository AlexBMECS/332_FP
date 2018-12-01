import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import sys
# from align import findTop, findBottom, findLeft, findRight

def findTop(img):
	# E = cv2.Canny(img,100,200)
	for i in range(img.shape[0]): # rows
		row = img[i,:]

		if len(row[row > 0]) > 0:
			return i


def findBottom(img):
	# E = cv2.Canny(img,100,200)
	for i in reversed(range(img.shape[0])):
		row = img[i,:]
		
		if len(row[row > 0]) > 0:
			return i


def findLeft(img):
	# E = cv2.Canny(img,100,200)
	for i in range(img.shape[1]):
		col = img[:,i]

		if len(col[col > 0]) > 0:
			return i


def findRight(img):
	# E = cv2.Canny(img,100,200)
	for i in reversed(range(img.shape[1])):
		col = img[:,i]

		if len(col[col > 0]) > 0:
			return i

def pad_with(vector, pad_width, iaxis, kwargs):
	pad_value = kwargs.get('padder', 10)
	vector[:pad_width[0]] = pad_value
	vector[-pad_width[1]:] = pad_value
	return vector


def pieceExtraction(CCL, img, store_files=True	):
	labels = np.unique(CCL)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
	closing = cv2.morphologyEx(CCL, cv2.MORPH_CLOSE, kernel)

	segmented_bw = []
	segmented_clr = []

	for label in labels:
		if label == 0:
			continue	

		segment_bw = closing == label

		segment_clr = np.zeros([segment_bw.shape[0], segment_bw.shape[1], 3])

		for i in range(segment_bw.shape[0]):
			for j in range(segment_bw.shape[1]):
				if segment_bw[i,j] != 0:
					segment_clr[i,j,0] = img[i,j,0]
					segment_clr[i,j,1] = img[i,j,1]
					segment_clr[i,j,2] = img[i,j,2]

		# segment_clr = segment_clr.astype('uint8')

		# subimage_bw = segment_bw[findTop(segment_bw)-50:findBottom(segment_bw)+50,
		# 		findLeft(segment_bw)-50:findRight(segment_bw)+50]

		subimage_bw = segment_bw[findTop(segment_bw):findBottom(segment_bw),
				findLeft(segment_bw):findRight(segment_bw)]


		subimage_clr = np.zeros([subimage_bw.shape[0], subimage_bw.shape[1], 3])


		subimage_clr[:,:,0] = segment_clr[findTop(segment_bw):findBottom(segment_bw),
				findLeft(segment_bw):findRight(segment_bw), 0]

		subimage_clr[:,:,1] = segment_clr[findTop(segment_bw):findBottom(segment_bw),
				findLeft(segment_bw):findRight(segment_bw), 1]

		subimage_clr[:,:,2] = segment_clr[findTop(segment_bw):findBottom(segment_bw),
				findLeft(segment_bw):findRight(segment_bw), 2]


		subimage_bw = np.pad(subimage_bw, 100, pad_with, padder=0)
		subimage_clr_0 = np.pad(subimage_clr[:,:,0], 100, pad_with, padder=0)
		subimage_clr_1 = np.pad(subimage_clr[:,:,1], 100, pad_with, padder=0)
		subimage_clr_2 = np.pad(subimage_clr[:,:,2], 100, pad_with, padder=0)

		subimage_clr = np.zeros([subimage_clr_0.shape[0], subimage_clr_0.shape[1], 3])
		subimage_clr[:,:,0] = subimage_clr_0
		subimage_clr[:,:,1] = subimage_clr_1
		subimage_clr[:,:,2] = subimage_clr_2

		subimage_clr = subimage_clr.astype('uint8')

		segmented_bw.append(subimage_bw)
		segmented_clr.append(subimage_clr)

	if store_files:
		for i in range(len(segmented_bw)):
			piece = segmented_bw[i].astype('uint8')	
			np.save('new_piece_%s' % i, piece)


		for i in range(len(segmented_clr)):
			piece = segmented_clr[i].astype('uint8')
			# plt.imshow(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
			# plt.show()
			np.save('new_piece_clr_%s' % i, piece)

	return 


def main():
	img = cv2.imread('squirrel.jpg')
	CCL = np.load('labeled_components')
	pieceExtraction(CCL, img)


if __name__ == "__main__":
	main()


