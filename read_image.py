import cv2

def image_read(input_file):

	im = cv2.imread(input_file)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

	return im