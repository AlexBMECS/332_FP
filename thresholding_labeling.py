# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 02:27:41 2018

@author: alexrazer
"""
import numpy as np
import cv2 as cv
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.image as mpimg
import math
import time
from scipy import ndimage

def thresholding_morph(filename):
    t = cv.imread(filename)
    #t =  t[750:1750,750:1750,:]
    tn = cv.cvtColor(t,cv.COLOR_BGR2GRAY)
    dim = tn.shape

    for x in range(dim[0]):
        for y in range(dim[1]):
            if(tn[x,y]>40):
                tn[x,y] = 255
            else:
                tn[x,y] = 0
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(20,25))
    closing = cv.morphologyEx(tn, cv.MORPH_CLOSE, kernel)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
    return opening


def CCL(img):
    def find_min(e_table, number):
        t = np.where(e_table[number] == 1)
        return t[0][0]

    #Pass 1
    img = img
    I = img[:,:]/(255)
    I = I.astype(int)

    dim = list(I.shape)
    newimg = np.zeros(dim)
    newimg = newimg.astype(int)
    e_table = np.identity(4500)
    L=1

    for row in range(dim[0]):
        for col in range(dim[1]):
            pixel = I[row,col]
            if(pixel == 1):
                upper = newimg[row-1,col]
                left = newimg[row,col-1]
                if(upper == 0):
                    if(left == 0):
                        newimg[row,col] = L
                        L+=1
                    else:
                        newimg[row,col] = max(left,upper)
                if(upper != 0):
                    if(left == 0):
                        newimg[row,col] = max(left,upper)
                    elif(left == upper):
                        newimg[row,col] = upper
                    else:
                        newimg[row,col] = min(left,upper)
                        e_table[max(left,upper)][min(left,upper)] = 1


    #Pass 2
    for u in range(dim[0],0,-1):
        t = np.where(e_table[u] == 1)
        p = t[0][0]
        newimg[np.where(newimg == u)] = p
        
    #Size Filtration
    uni = []
    for u, row in enumerate(newimg):
        for v, pixel in enumerate(row):
            if pixel not in uni and pixel != 0:
                uni.append(pixel)

    u,counts = np.unique(newimg, return_counts = True)
    ucount = np.asarray((u, counts)).T

    remove = []
    for idx, count in enumerate(counts):
        if count < 10:
            remove.append(u[idx])

    for u, row in enumerate(newimg):
        for v, pixel in enumerate(row):
            if pixel != 0 and pixel in remove:
                newimg[u][v] = 0

    for rowID in range(len(img)):
        for colID in range(len(img[0])):
            img[rowID][colID] = (newimg[rowID][colID]*70)%255
            

    plt.imshow(img)
    plt.show()
    return newimg,len(np.unique(newimg)-1)
