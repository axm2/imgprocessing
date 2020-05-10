import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms


#open the image
underexposed = cv2.imread("input/underexposed.jpg")
overexposed = cv2.imread("input/overexposed.jpg")

#grayscale
gray_underexposed = cv2.cvtColor(underexposed, cv2.COLOR_BGR2GRAY)
gray_overexposed = cv2.cvtColor(overexposed, cv2.COLOR_BGR2GRAY)

#equalize histogram of input image
equ_un = cv2.equalizeHist(gray_underexposed)
equ_over = cv2.equalizeHist(gray_overexposed)

plt.hist(equ_un.ravel(), 256, [0, 256])
plt.show()
plt.hist(equ_over.ravel(), 256, [0, 256])
plt.show()

#reference
data = np.random.uniform(0, 1, 100)
count, bins, ignored = plt.hist(data, facecolor='blue')

#match histograms
matchedUnder = match_histograms(equ_un, count, multichannel=True)
matchedOver = match_histograms(equ_over, count, multichannel=True)
cv2.imwrite("matchUnder.png", matchedUnder)
cv2.imwrite("matchUnder.png", matchedUnder)