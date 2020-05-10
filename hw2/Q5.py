import cv2
from IPython.display import Image

# a. Smoothing spatial filtering (Gaussian and Box Kernels)

# Gaussian Blur
img = cv2.imread("input/overexposed.jpg")
x = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imwrite('output/Q5_GaussianBlur.jpg', x)
Image('output/Q5_GaussianBlur.jpg')

# Box Kernels
img = cv2.imread("input/overexposed.jpg")
x = cv2.boxFilter(img, -1, (9, 9))
cv2.imwrite('output/Q5_BoxKernels.jpg', x)
Image('output/Q5_BoxKernels.jpg')

# b. First-order derivative (Robert and Sobel Kernels)

# Robert Kernel
x = cv2.Laplacian(img, cv2.CV_64F)
cv2.imwrite('output/Q5_RobertKernel.jpg', x)
Image('output/Q5_RobertKernel.jpg')

# Sobel Kernel
x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
cv2.imwrite('output/Q5_SobelKernel.jpg', x)
Image('output/Q5_SobelKernel.jpg')

# c. Second-order derivative

# Second Derivative
img = cv2.imread('output/Q5_RobertKernel.jpg')
x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
cv2.imwrite('output/Q5_SecondDerivative.jpg', x)
Image('output/Q5_SecondDerivative.jpg')

# d. Unsharp and Highboost filtering

# Unsharp and Highboost Filtering
gauss = cv2.GaussianBlur(img, (7, 7), 0)
unsharp_image = cv2.addWeighted(img, 12, gauss, -1, 0)
cv2.imwrite('output/Q5_Sharpened.jpg', unsharp_image)
Image('output/Q5_Sharpened.jpg')