import thresholding_labeling
from thresholding_labeling import CCL
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import math

test = CCL(thresholding_labeling.thresholding_morph('all_pieces.jpg'))

np.save('ccl_out', test[0])

plt.imshow(test[0])
plt.show()