import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from align import findTop, findBottom, findLeft, findRight

img = cv2.imread('squirrel.jpg')

CCL = np.loadtxt('labeled_components.csv', delimiter=',')
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
			if segment_bw[i,j]:
				segment_clr[i,j,0] = img[i,j,0]
				segment_clr[i,j,1] = img[i,j,1]
				segment_clr[i,j,2] = img[i,j,2]


	subimage_bw = segment_bw[findTop(segment_bw)-100:findBottom(segment_bw)+100,
 		findLeft(segment_bw)-100:findRight(segment_bw)+100]

 	# segment_bw[findTop(segment_bw)-100:findBottom(segment_bw)+100,
 	# 		findLeft(segment_bw)-100:findRight(segment_bw)+100]

	# subimage_clr = img[segment_bw,:]
	subimage_clr = np.zeros([subimage_bw.shape[0], subimage_bw.shape[1], 3])


	subimage_clr[:,:,0] = segment_clr[findTop(segment_bw)-100:findBottom(segment_bw)+100,
			findLeft(segment_bw)-100:findRight(segment_bw)+100, 0]

	subimage_clr[:,:,1] = segment_clr[findTop(segment_bw)-100:findBottom(segment_bw)+100,
			findLeft(segment_bw)-100:findRight(segment_bw)+100, 1]

	subimage_clr[:,:,2] = segment_clr[findTop(segment_bw)-100:findBottom(segment_bw)+100,
			findLeft(segment_bw)-100:findRight(segment_bw)+100, 2]


	plt.imshow(subimage_clr)
	plt.show()
	exit()


	segmented_bw.append(subimage_bw)
	segmented_clr.append(segment_clr)


plt.imshow(segmented_bw[0])
plt.show()

# subimage = segmented[0][findTop(segmented[0])-100:findBottom(segmented[0])+100,
# 			findLeft(segmented[0])-100:findRight(segmented[0])+100]

# plt.subplot(221)
# plt.imshow(segmented[0])
# plt.subplot(222)
# plt.imshow(segmented[1])
# plt.subplot(223)
# plt.imshow(segmented[2])
# plt.subplot(224)
# plt.imshow(segmented[3])
# plt.show()