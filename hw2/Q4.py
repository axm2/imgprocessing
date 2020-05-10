import cv2
from matplotlib import pyplot as plt
import numpy as np

img1 = cv2.imread("input/overexposed.jpg")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("input/underexposed.jpg")
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Kernels
kernel_w1 = 1.0 * np.array([[1]])
kernel_w2 = float(1 / 5) * np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
kernel_w3 = float(1 / 9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

# Apply the kernels to the images
img1_w1 = cv2.filter2D(gray1, -1, kernel_w1)
img1_w2 = cv2.filter2D(gray1, -1, kernel_w2)
img1_w3 = cv2.filter2D(gray1, -1, kernel_w3)

img2_w1 = cv2.filter2D(gray2, -1, kernel_w1)
img2_w2 = cv2.filter2D(gray2, -1, kernel_w2)
img2_w3 = cv2.filter2D(gray2, -1, kernel_w3)

# Save Output
cv2.imwrite('output/Q4_img1_kernel1.jpg', img1_w1)
cv2.imwrite('output/Q4_img1_kernel2.jpg', img1_w2)
cv2.imwrite('output/Q4_img1_kernel3.jpg', img1_w3)

cv2.imwrite('output/Q4_img2_kernel1.jpg', img2_w1)
cv2.imwrite('output/Q4_img2_kernel2.jpg', img2_w2)
cv2.imwrite('output/Q4_img2_kernel3.jpg', img2_w3)

# Histogram
img1_w1_hist = cv2.calcHist([img1_w1], [0], None, [256], [0, 256])
img1_w2_hist = cv2.calcHist([img1_w2], [0], None, [256], [0, 256])
img1_w3_hist = cv2.calcHist([img1_w3], [0], None, [256], [0, 256])

img2_w1_hist = cv2.calcHist([img2_w1], [0], None, [256], [0, 256])
img2_w2_hist = cv2.calcHist([img2_w2], [0], None, [256], [0, 256])
img2_w3_hist = cv2.calcHist([img2_w3], [0], None, [256], [0, 256])

# Plot
plt.plot(img1_w1_hist)
plt.title('Overexposed - w1')
plt.savefig('output/Q4_img1_kernel1_hist.png')
plt.show()

plt.plot(img1_w2_hist)
plt.title('Overexposed - w2')
plt.savefig('output/Q4_img1_kernel2_hist.png')
plt.show()

plt.plot(img1_w3_hist)
plt.title('Overexposed - w3')
plt.savefig('output/Q4_img1_kernel3_hist.png')
plt.show()

plt.plot(img2_w1_hist)
plt.title('Underexposed - w1')
plt.savefig('output/Q4_img2_kernel1_hist.png')
plt.show()

plt.plot(img2_w2_hist)
plt.title('Underexposed - w2')
plt.savefig('output/Q4_img2_kernel2_hist.png')
plt.show()

plt.plot(img2_w3_hist)
plt.title('Underexposed - w3')
plt.savefig('output/Q4_img2_kernel3_hist.png')
plt.show()