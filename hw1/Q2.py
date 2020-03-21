import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter
from PIL import Image

img1 = cv2.imread('input/uniform_scene1.jpg')
img2 = cv2.imread('input/uniform_scene2.jpg')

img3 = img1 - img2

# cv2.cvtColor is applied over the
# image input with applied parameters
# to convert the image in grayscale
img = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# applying thresholding
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

# Save output img
cv2.imwrite('output/Q2_binary_threshold.png', thresh1)

histr = cv2.calcHist([thresh1], [0], None, [256], [0, 256])

# show the plotting graph of an image
plt.plot(histr)
#Save histogram
plt.savefig('output/Q2_histogram.png')
plt.show()

# Quantify number of black pixels

# get all non black Pixels
cntNotBlack = cv2.countNonZero(thresh1)

# get pixel count of image
height, width = thresh1.shape
cntPixels = height*width

# compute all black pixels
cntBlack = cntPixels - cntNotBlack

print("Total Number of Pixels: ", cntPixels)
print("Total Number of Non-Black Pixels: ", cntNotBlack)
print("Total Number of Black Pixels: ", cntBlack)


cv2.imshow('Binary Threshold', thresh1)
cv2.waitKey(0)
