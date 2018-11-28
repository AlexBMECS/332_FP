import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from align import findTop, findBottom, findLeft, findRight

read_csv = np.loadtxt('labeled_components.csv', delimiter=',')
labels = np.unique(read_csv)

blank_canvas = np.zeros(read_csv.shape)
segmented = []

for label in labels:
	if label == 0:
		continue	

	# segment = read_csv == label

	segmented.append(read_csv == label)

subimage = segmented[0][findTop(segmented[0])-100:findBottom(segmented[0])+100,
			findLeft(segmented[0])-100:findRight(segmented[0])+100]


plt.imshow(subimage)
plt.show()