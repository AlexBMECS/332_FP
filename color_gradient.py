import cv2
import numpy as np


def colorError(pixel_a, pixel_b):
	r_1 = pixel_a[2]
	g_1 = pixel_a[1]
	b_1 = pixel_a[0]

	r_2 = pixel_b[2]
	g_2 = pixel_b[1]
	b_2 = pixel_b[0]

	distance = np.sqrt((r_2-r_1)**2 + (g_2-g_1)**2 + (b_2-b_1)**2)
	return distance


def average_error(region_1, region_2):
	'''
	Assume input of two numpy nd-arrays

	Take average of each row (across columns). Assume comparisons made row-by-row.
	Will have # pixels averaged corresponding to # rows.

	'''
	val_1 = np.zeros([3])
	val_2 = np.zeros([3])


	val_1[0] = np.mean(region_1[:,:,0])
	val_1[1] = np.mean(region_1[:,:,1])
	val_1[2] = np.mean(region_1[:,:,2])

	val_2[0] = np.mean(region_2[:,:,0])
	val_2[1] = np.mean(region_2[:,:,1])
	val_2[2] = np.mean(region_2[:,:,2])

	colorError(val_1, val_2)

	# for i in range(40):
	# 	pixel_1 = row_1[i]
	# 	pixel_2 = row_2[i]

	# 	err = colorError(pixel_1, pixel_2)
	# 	total_err += err

	# return total_err / 40


average_error()


