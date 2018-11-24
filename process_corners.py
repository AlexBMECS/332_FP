import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def process_corners(corners):
	min_mag = sys.maxsize
	min_index = 0
	i = 0

	for corner in corners:
		mag = np.sqrt(corner[0]**2 + corner[1]**2)
		
		if mag < min_mag:
			min_mag = mag
			min_index = i

		i += 1

	corners[min_index], corners[0] = corners[0], corners[min_index]

	x = corners[0][0]
	y = corners[0][1]

	for i in range(len(corners)):
		corners[i] = (corners[i][0] - x, corners[i][1] - y)

	c1 = corners[0]
	c2 = corners[1]
	c4 = corners[3]

	dot_product = c2[0]*c4[0] + c2[1]*c4[1]

	if np.absolute(dot_product) > 0.01:
		mag_2 = np.sqrt(c2[0]**2 + c2[1]**2)
		mag_4 = np.sqrt(c4[0]**2 + c4[1]**2)

		if mag_2 < mag_4:
			corners[2], corners[3] = corners[3], corners[2]

		else:
			corners[2], corners[1] = corners[1], corners[2]

	else:
		print('done')

	for i in range(len(corners)):
		corners[i] = (corners[i][0] + x, corners[i][1] + y)

	return corners


def get_avg_angle(corners):
	
	p1x = corners[0][0]
	p1y = corners[0][1]

	p2x = corners[1][0]
	p2y = corners[1][1]

	p3x = corners[2][0]
	p3y = corners[2][1]

	p4x = corners[3][0]
	p4y = corners[3][1]

	inner1 = (p4x-p1x)*(p2x-p1x) + (p4y-p1y)*(p2y-p1y)
	norm2 = np.hypot(p2x - p1x, p2y - p1y)
	norm4 = np.hypot(p4x - p1x, p4y - p1y)
	angle1 = np.arccos(inner1 / (norm2*norm4))

	inner2 = (p3x-p2x)*(p1x-p2x) + (p1y-p2y)*(p3y-p2y)
	norm1 = np.hypot(p1x - p2x, p1y - p2y)
	norm3 = np.hypot(p3x - p2x, p3y - p2y)
	angle2 = np.arccos(inner2 / (norm1*norm3))

	inner3 = (p4x-p3x)*(p2x-p3x) + (p4y-p3y)*(p2y-p3y)
	norm2 = np.hypot(p2x - p3x, p2y - p3y)
	norm4 = np.hypot(p4x - p3x, p4y - p3y)
	angle3 = np.arccos(inner3 / (norm2*norm4))

	inner4 = (p3x-p4x)*(p1x-p4x) + (p1y-p4y)*(p3y-p4y)
	norm1 = np.hypot(p1x - p4x, p1y - p4y)
	norm3 = np.hypot(p3x - p4x, p3y - p4y)
	angle4 = np.arccos(inner4 / (norm1*norm3))

	avg_angle = (angle1 + angle2 + angle3 + angle4) / 4

	return avg_angle


corn1 = (1, 2)
corn2 = (2, 2)
corn3 = (1, 3)
corn4 = (2, 3)
corners = [corn1, corn2, corn3, corn4]

output = process_corners(corners)
print(output)
get_avg_angle(output)

plt.scatter(*zip(*output))
plt.show()
