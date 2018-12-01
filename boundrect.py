import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def rotate(img):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-45,1)
    dst = cv2.warpAffine(img,M,(cols,rows))  
        
    return dst


img = cv2.imread('images.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, im_bw) = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
im_bw[im_bw > 0] = 1
im_bw[im_bw == 0] = 255
im_bw[im_bw == 1] = 0 
rotato = rotate(im_bw)
rotato[rotato > 0] = 1
rotato[rotato == 0] = 255
rotato[rotato == 1] = 0 
ret, thresh = cv2.threshold(rotato,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,1,2)
cnt = contours[0]

# Basic box: should only work with aligned image
'''
x,y,w,h = cv2.boundingRect(cnt)
print((x,y,w,h))
out = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('output',cv2.resize(out, (int(out.shape[0]*0.5),int(out.shape[1]*0.5))))
cv2.waitKey(0)
'''

# Draws contoours around non-aligned image
'''
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
out = cv2.drawContours(im2,[box],0,(0,0,255),2)
cv2.imshow('image',cv2.resize(out, (int(out.shape[0]*0.5),int(out.shape[1]*0.5))))
cv2.waitKey(0)
'''