import cv2

def edge_detection(im, low, high):

	final = cv2.Canny(im, low, high)

	return final