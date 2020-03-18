import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter

def histogram_maker(channel, title):
    E = Counter(channel.flatten()) # working as intended
    D = OrderedDict(sorted(E.items()))
    plt.bar(range(len(D)), list(D.values()), align='center')
    plt.xlim(right=255)
    plt.title(title)
    plt.xlabel('pixel intensity')
    plt.ylabel('number of pixels')
    plt.show()

#q1 use opencv, numpy, matplot
overexposed_img = cv2.imread('E:/Documents/imgprocessing/hw1/overexposed.jpg')
underexposed_img = cv2.imread('E:/Documents/imgprocessing/hw1/underexposed.jpg')
img = cv2.imread('E:/Documents/bighead.jpg')
red_channel = img[:,:,2]
green_channel = img[:,:,1]
blue_channel = img[:,:,0]
histogram_maker(red_channel, 'red channel histogram')
histogram_maker(green_channel, 'green channel histogram')
histogram_maker(blue_channel, 'blue channel histogram')