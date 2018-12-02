import cv2
import numpy as np


def color_error(pixel_a, pixel_b):
	'''
	Euclidian distance error caluclation between two pixels
	'''
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
	Take average of each row (across columns). Assume comparisons made row-by-row.
	Will have # pixels averaged corresponding to # rows.
	'''

	# Transpose ndarrays to check for correct shape.

	region_1 = region_1.astype('float64')
	region_2 = region_2.astype('float64')

	if region_1.shape[0] < region_1.shape[1]:
		region_1 = region_1.swapaxes(0,1)

	if region_2.shape[0] < region_2.shape[1]:
		region_2 = region_2.swapaxes(0,1)


	avg_1 = np.mean(region_1, axis=1)
	avg_2 = np.mean(region_2, axis=1)

	err = 0

	# Compute error for each averaged row
	for i in range(avg_1.shape[0]):
		pixel_1 = avg_1[i,:]
		pixel_2 = avg_2[i,:]

		err += color_error(pixel_1, pixel_2)

	return err / avg_1.shape[0]


def main():
	a1 = np.zeros([5, 20, 3])
	a2 = np.zeros([20, 5, 3])
	avg = average_error(a1,a2)
	print(avg)


if __name__ == "__main__":
	main()

