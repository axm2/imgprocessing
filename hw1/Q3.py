
from spectral import *

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

img = open_image('TIPJUL1.lan')
img.__class__
rgb_image = np.zeros((img.shape[0],img.shape[1],3),np.int32)
rgb_image[:,:,0] = np.reshape(img[:,:,3],(169,169))
rgb_image[:,:,1] = np.reshape(img[:,:,2],(169,169))
rgb_image[:,:,2] = np.reshape(img[:,:,1],(169,169))
plt.figure(1)
plt.imshow(rgb_image)
plt.title("False Color Image")
plt.savefig('FalseColor.png')
ndvi_image = (img[:,:,3]-img[:,:,1])/(img[:,:,3]+img[:,:,1])
plt.figure(2)
plt.imshow(ndvi_image.reshape(169,169), cmap=plt.cm.gray)
plt.title("NDVI Image")
#plt.show()
plt.savefig('NDVI.png')