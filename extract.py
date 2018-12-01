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


def pieceExtraction(CCL, img, store_files=True):
	labels = np.unique(CCL)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
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

		subimage_bw = segment_bw[findTop(segment_bw)-100:findBottom(segment_bw)+100,
	 		findLeft(segment_bw)-100:findRight(segment_bw)+100]

		subimage_clr = np.zeros([subimage_bw.shape[0], subimage_bw.shape[1], 3])


		subimage_clr[:,:,0] = segment_clr[findTop(segment_bw)-100:findBottom(segment_bw)+100,
				findLeft(segment_bw)-100:findRight(segment_bw)+100, 0]

		subimage_clr[:,:,1] = segment_clr[findTop(segment_bw)-100:findBottom(segment_bw)+100,
				findLeft(segment_bw)-100:findRight(segment_bw)+100, 1]

		subimage_clr[:,:,2] = segment_clr[findTop(segment_bw)-100:findBottom(segment_bw)+100,
				findLeft(segment_bw)-100:findRight(segment_bw)+100, 2]

		subimage_clr = subimage_clr.astype('uint8')

		# plt.imshow(cv2.cvtColor(subimage_clr, cv2.COLOR_BGR2RGB))
		# plt.show()

		segmented_bw.append(subimage_bw)
		segmented_clr.append(subimage_clr)

	if store_files:
		for i in range(len(segmented_bw)):
			piece = segmented_bw[i].astype('uint8')	
			np.save('piece_%s' % i, piece)


		for i in range(len(segmented_clr)):
			piece = segmented_clr[i].astype('uint8')
			# plt.imshow(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
			# plt.show()
			np.save('piece_clr_%s' % i, piece)

	return 


def main():
	img = cv2.imread('squirrel.jpg')

	print(img[0,0].shape)
	exit()
	CCL = np.loadtxt('labeled_components.csv', delimiter=',')
	pieceExtraction(CCL, img)


if __name__ == "__main__":
	main()


