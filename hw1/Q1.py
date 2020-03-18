import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter

def histogram_maker(channel, title, ax=None):
    ax = ax or plt.gca()
    E = Counter(channel.flatten()) # working as intended
    D = OrderedDict(sorted(E.items()))
    line = ax.bar(range(len(D)), list(D.values()), align='center')
    ax.set_xlim(right=255)
    ax.set_title(title)
    ax.set_xlabel('pixel intensity')
    ax.set_ylabel('number of pixels')
    return line

#q1 use opencv, numpy, matplot
fig, ((ax1, ax2, ax3), (bx1, bx2, bx3)) = plt.subplots(2, 3, sharex='all', sharey='all')
overexposed_img = cv2.imread('E:/Documents/imgprocessing/hw1/overexposed.jpg')
underexposed_img = cv2.imread('E:/Documents/imgprocessing/hw1/underexposed.jpg')
img = cv2.imread('E:/Documents/bighead.jpg')
red_channel = img[:,:,2]
green_channel = img[:,:,1]
blue_channel = img[:,:,0]
histogram_maker(red_channel, 'red channel histogram', ax1)
histogram_maker(green_channel, 'green channel histogram', ax2)
histogram_maker(blue_channel, 'blue channel histogram', ax3)
histogram_maker(red_channel, 'red channel histogram', bx1)
histogram_maker(green_channel, 'green channel histogram', bx2)
histogram_maker(blue_channel, 'blue channel histogram', bx3)
plt.show()