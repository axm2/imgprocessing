import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter

img1 = cv2.imread('uniform_scene1.jpg')
img2 = cv2.imread('uniform_scene2.jpg')

img3 = img1 - img2

# cv2.cvtColor is applied over the
# image input with applied parameters
# to convert the image in grayscale
img = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# applying thresholding
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

histr = cv2.calcHist([img], [0], None, [256], [0, 256])

# show the plotting graph of an image
plt.plot(histr)
plt.show()

cv2.imshow('Binary Threshold', thresh1)
cv2.waitKey(0)