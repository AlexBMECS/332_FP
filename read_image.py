import cv2

def image_read(input_file):

	im = cv2.imread(input_file)

	return im