import cv2
import numpy as np 
import matplotlib.pyplot as plt
import math
import itertools
from rotate import rotate

def label_sides(img, piece, clrpiece):

	# plt.imshow(img)
	# plt.title('{} pre-rotated image'.format(piece))
	# plt.figure()

	if piece == 'piece_5':
		interval = [1.4, 1.7]
	elif piece == 'piece_11':
		interval = [1.2, 2.1]
	else:
		interval = [1.25, 2.1]

	corners = find_corners(img, interval)
	# plt.imshow(img)
	# for corner in corners:
	# 	plt.plot(corner[0], corner[1], 'ro')
	# plt.title('corners from find_corners using box')
	# plt.show()
	x_cen = [p[0] for p in corners]
	y_cen = [p[1] for p in corners]
	center = (sum(x_cen) / len(corners), sum(y_cen) / len(corners))

	# plt.plot(center[0], center[1], 'ro')
	# plt.imshow(img)
	# plt.title('new center created from corners')
	# plt.show()

	angle = math.degrees(find_offset(img, center))
	#print('angle', angle)
	img = rotate(img, angle)
	# plt.imshow(img)
	# plt.title('{} rotated image'.format(piece))
	# plt.figure()
	corners = find_corners(img, interval)
	new_corners = []
	for corner in corners:
		corner[0] = int(corner[0])
		corner[1] = int(corner[1])
	#print(img)

	#plt.plot(center[0], center[1], 'ro')

	def midpoint(p1, p2):
		return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

	#Labeling sides
	edge_shape = []
	side_lengths = []
	i = 0
	while i < len(corners):

		if i == 0:
			midp = midpoint(corners[i], corners[i+1])
			x1 = corners[i][0]
			x2 = corners[i+1][0]
			y1 = corners[i][1]
			y2 = corners[i+1][1]
			dist = math.hypot(x2 - x1, y2 - y1)
			side_lengths.append(dist)
			x, y = midp
			x = int(x)
			y = int(y)
			shape = 0
			if img[y - 10][x] > 0 and img[y + 10][x] == 0:
				plt.plot(x, y, 'ro')
				plt.text(x * (1 + 0.01), y * (1 + 0.01) , 'flat', fontsize=12, color='white')
				shape = 0
			elif img[y - 10][x] > 0 and img[y + 10][x] > 0:
				plt.plot(x, y, 'ro')
				plt.text(x * (1 + 0.01), y * (1 + 0.01) , 'head', fontsize=12, color='white')
				shape = 1
			elif img[y - 10][x] == 0 and img[y + 10][x] == 0:
				plt.plot(x, y, 'ro')
				plt.text(x * (1 + 0.01), y * (1 + 0.01) , 'hole', fontsize=12, color='white')
				shape = -1
			edge_shape.append(shape)
		elif i == 1:
			midp = midpoint(corners[i], corners[i+1])
			x1 = corners[i][0]
			x2 = corners[i+1][0]
			y1 = corners[i][1]
			y2 = corners[i+1][1]
			dist = math.hypot(x2 - x1, y2 - y1)
			side_lengths.append(dist)
			x, y = midp
			x = int(x)
			y = int(y)
			shape = 0
			if img[y][x+10] > 0 and img[y][x - 10] == 0:
				plt.plot(x, y, 'ro')
				plt.text(x * (1 + 0.01), y * (1 + 0.01) , 'flat', fontsize=12, color='white')
				shape = 0
			elif img[y][x+10] > 0 and img[y][x - 10] > 0:
				plt.plot(x, y, 'ro')
				plt.text(x * (1 + 0.01), y * (1 + 0.01) , 'head', fontsize=12, color='white')
				shape = 1
			elif img[y][x+10] == 0 and img[y][x - 10] == 0:
				plt.plot(x, y, 'ro')
				plt.text(x * (1 + 0.01), y * (1 + 0.01) , 'hole', fontsize=12, color='white')
				shape = -1
			edge_shape.append(shape)
		elif i == 2:
			midp = midpoint(corners[i], corners[i+1])
			x1 = corners[i][0]
			x2 = corners[i+1][0]
			y1 = corners[i][1]
			y2 = corners[i+1][1]
			dist = math.hypot(x2 - x1, y2 - y1)
			side_lengths.append(dist)
			x, y = midp
			x = int(x)
			y = int(y)
			shape = 0
			if img[y + 10][x] > 0 and img[y - 10][x] == 0:
				plt.plot(x, y, 'ro')
				plt.text(x * (1 + 0.01), y * (1 + 0.01) , 'flat', fontsize=12, color='white')
				shape = 0
			elif img[y + 10][x] > 0 and img[y - 10][x] > 0:
				plt.plot(x, y, 'ro')
				plt.text(x * (1 + 0.01), y * (1 + 0.01) , 'head', fontsize=12, color='white')
				shape = 1
			elif img[y + 10][x] == 0 and img[y - 10][x] == 0:
				plt.plot(x, y, 'ro')
				plt.text(x * (1 + 0.01), y * (1 + 0.01) , 'hole', fontsize=12, color='white')
				shape = -1
			edge_shape.append(shape)
		else:
			#i == 3:
			midp = midpoint(corners[0], corners[3])
			x1 = corners[0][0]
			x2 = corners[3][0]
			y1 = corners[0][1]
			y2 = corners[3][1]
			dist = math.hypot(x2 - x1, y2 - y1)
			side_lengths.append(dist)
			x, y = midp
			x = int(x)
			y = int(y)
			shape = 0
			if img[y][x-10] > 0 and img[y][x + 10] == 0:
				plt.plot(x, y, 'ro')
				plt.text(x * (1 + 0.01), y * (1 + 0.01) , 'flat', fontsize=12, color='white')
				shape = 0
			elif img[y][x-10] > 0 and img[y][x + 10] > 0:
				plt.plot(x, y, 'ro')
				plt.text(x * (1 + 0.01), y * (1 + 0.01) , 'head', fontsize=12, color='white')
				shape = 1
			elif img[y][x-10] == 0 and img[y][x + 10] == 0:
				plt.plot(x, y, 'ro')
				plt.text(x * (1 + 0.01), y * (1 + 0.01) , 'hole', fontsize=12, color='white')
				shape = -1
			edge_shape.append(shape)

	    # if i == 3:
	    #     midp = midpoint(corners[0], corners[3])
	    #     x, y = midp
	    #     plt.plot(x, y, 'ro')
	    #     plt.text(x * (1 + 0.01), y * (1 + 0.01) , i, fontsize=12)
	    #     break

	    # midp = midpoint(corners[i], corners[i+1])
	    # x, y = midp
	    # plt.plot(x, y, 'ro')
	    # plt.text(x * (1 + 0.01), y * (1 + 0.01) , i, fontsize=12)
		i += 1

	img = np.load(clrpiece)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = rotate(img, angle)
	plt.imshow(img)
	plt.show()

	return corners, edge_shape, side_lengths, img

def polarize(img, center):

    source = cv2.Canny(img, 100, 200)
    source[source > 0] = 255

    rhos = []
    thetas = []
    for rowIdx, row in enumerate(source):
        for colIdx, pixel in enumerate(row):
            if pixel > 0:

                x = colIdx - center[0]
                y = rowIdx - center[1]

                rho = math.sqrt(x**2 + y**2)

                if colIdx == center[0]:
                    if y > 0:
                        theta = math.pi/2
                    else:
                        theta = 3*math.pi/2
                else:
                    theta = math.atan(y/x)

                if x < 0:
                    theta += math.pi
                if theta < 0:
                    theta += 2*math.pi

                if theta in thetas:
                    index = thetas.index(theta)
                    if rho > rhos[index]:
                        thetas[index] = theta
                        rhos[index] = rho
                        continue
                    else:
                        continue

                rhos.append(rho)
                thetas.append(theta)

    tups = list(zip(thetas, rhos))
    tups.sort()
    thetas, rhos = map(list, zip(*tups))

    return thetas, rhos

def find_offset(img, center):

	thetas, rhos = polarize(img, center)

	#Finding local maximums (corners)
	local_max_indices = []
	local_maxes = []
	n = 13
	for idx, rho in enumerate(rhos):
	    if idx < n or idx > len(rhos) - n+1:
	        continue

	    if max(rhos[(idx - n):idx]) < rho and max(rhos[(idx + 1):(idx + n+1)]) < rho:
	        local_max_indices.append(idx)
	        local_maxes.append(thetas[idx])

	indices = list(range(0, len(local_maxes)))
	combos = list(itertools.combinations(indices, 2))
	true_indexes = []

	#print('find_offset local maxes', local_maxes)
	for combo in combos:
	    if local_maxes[combo[1]] - local_maxes[combo[0]] > 1.4 and local_maxes[combo[1]] - local_maxes[combo[0]] < 1.9:
	        if combo[0] not in true_indexes:
	            true_indexes.append(combo[0])
	        if combo[1] not in true_indexes:
	            true_indexes.append(combo[1])

	corners = []
	for index in true_indexes:
	    theta = thetas[local_max_indices[index]]
	    rho = rhos[local_max_indices[index]]
	    corners.append((theta, rho))

	if len(corners) == 3:
	    first_best_rho = max(rhos[:20])
	    index_rho = rhos.index(first_best_rho)
	    theta = thetas[index_rho]
	    corners.append((theta, first_best_rho))

	# plt.scatter(thetas, rhos)
	# for corner in corners:
	# 	plt.plot(corner[0], corner[1], 'ro')
	# plt.title('find_offset polar plot')
	# plt.show()

	offset = corners[0][0] - (math.pi / 4)

	# if offset > 0:
	# 	offset = -offset

	# print(offset)

	# plt.scatter(thetas, rhos)
	# plt.scatter(list(np.asarray(thetas) + offset), rhos)
	# plt.show()
	return offset

def find_corners(img, interval):

	#Creating box around puzzle piece to find centroid.
	ret,thresh = cv2.threshold(img,127,255,0)
	im2, contours,hierarchy = cv2.findContours(thresh, 1, 2)
	cnt = contours[0]
	rect = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(rect)
	x_cen = [p[0] for p in box]
	y_cen = [p[1] for p in box]
	center = (sum(x_cen) / len(box), sum(y_cen) / len(box))

	# plt.plot(center[0], center[1], 'ro')
	# plt.imshow(img)
	# plt.title('center created from box in find_corners')
	# plt.show()

	thetas, rhos = polarize(img, center)

	# plt.scatter(thetas, rhos)
	# plt.title('theta vs rho finding_corners with bad center')
	# plt.show()

	#Finding local maximums (corners)
	local_max_indices = []
	local_maxes = []
	n = 13
	for idx, rho in enumerate(rhos):
	    if idx < n or idx > len(rhos) - n+1:
	        continue

	    if max(rhos[(idx - n):idx]) < rho and max(rhos[(idx + 1):(idx + n+1)]) < rho:
	        local_max_indices.append(idx)
	        local_maxes.append(thetas[idx])

	indices = list(range(0, len(local_maxes)))
	combos = list(itertools.combinations(indices, 2))
	true_indexes = []

	#print('find_corners local maxes', local_maxes)
	for combo in combos:
	    if local_maxes[combo[1]] - local_maxes[combo[0]] > interval[0] and local_maxes[combo[1]] - local_maxes[combo[0]] < interval[1]:
	        if combo[0] not in true_indexes:
	            true_indexes.append(combo[0])
	        if combo[1] not in true_indexes:
	            true_indexes.append(combo[1])

	corners = []
	for index in true_indexes:
	    theta = thetas[local_max_indices[index]]
	    rho = rhos[local_max_indices[index]]
	    corners.append((theta, rho))

	if len(corners) == 3:
	    first_best_rho = max(rhos[:50])
	    index_rho = rhos.index(first_best_rho)
	    theta = thetas[index_rho]
	    corners.append((theta, first_best_rho))

	# plt.scatter(thetas, rhos)
	# for corner in corners:
	# 	plt.plot(corner[0], corner[1], 'ro')
	# plt.title('theta vs rho finding_corners with bad center')
	# plt.show()

	xy_corners = []
	for corner in corners:
	    theta = corner[0]
	    rho = corner[1]

	    x = rho*math.cos(theta) + center[0]
	    y = rho*math.sin(theta) + center[1]

	    xy_corners.append([x,y])

	return xy_corners

#piece 5 findcorners: 1.4<x<1.7

#best parameters: findoffset: n = 13, 1.4<x<1.9, findcorners: n=13, 1.25<x<2.1