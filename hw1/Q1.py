import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter

def histogram_maker(channel, color, ax=None):
    ax = ax or plt.gca()
    E = Counter(channel.flatten()) # working as intended
    D = OrderedDict(sorted(E.items()))
    line = ax.bar(range(len(D)), list(D.values()), align='center', color=color)
    ax.set_xlim(right=255)
    ax.set_title(color + ' channel histogram')
    ax.set_xlabel('pixel intensity')
    ax.set_ylabel('number of pixels')
    #ax.set_color(color)
    return line

def channel_splitter(img, x1, x2, x3, x4):
    red_channel = img[:,:,2]
    green_channel = img[:,:,1]
    blue_channel = img[:,:,0]
    gray_channel = (red_channel + green_channel + blue_channel) / 3
    histogram_maker(red_channel, 'red', x1)
    histogram_maker(green_channel, 'green', x2)
    histogram_maker(blue_channel, 'blue', x3)
    histogram_maker(gray_channel, 'gray', x4)
    return

#q1 use opencv, numpy, matplot
plt.style.use('dark_background')
fig, ((ax1, ax2, ax3, ax4), (bx1, bx2, bx3, bx4)) = plt.subplots(2, 4, sharex='all', sharey='all')
fig.suptitle('Overexposed image on top VS Underexposed image below')
overexposed_img = cv2.imread('input/overexposed.jpg')
underexposed_img = cv2.imread('input/underexposed.jpg')
channel_splitter(overexposed_img, ax1, ax2, ax3, ax4)
channel_splitter(underexposed_img, bx1, bx2, bx3, bx4)
plt.savefig('output/Q1_histogram.png')