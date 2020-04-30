import cv2
import numpy as np
from matplotlib import pyplot as plt

# Open the image.
underexposed = cv2.imread("hw2/input/underexposed.jpg")
overexposed = cv2.imread("hw2/input/overexposed.jpg")
underexposed = cv2.resize(underexposed, (1000, 750))
overexposed = cv2.resize(overexposed, (1000, 750))

#make them grayscale
gray_underexposed = cv2.cvtColor(underexposed, cv2.COLOR_BGR2GRAY)
gray_overexposed = cv2.cvtColor(overexposed, cv2.COLOR_BGR2GRAY)

#apply equalization
equalized_u = cv2.equalizeHist(gray_underexposed)
equalized_o = cv2.equalizeHist(gray_overexposed)

#place original and equalized side by side
res_o = np.hstack((gray_overexposed, equalized_o))
res_u = np.hstack((gray_underexposed, equalized_u))

#show images
cv2.imshow("overexposed original vs equalized", res_o)
cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image
cv2.imshow("underexposed original vs equalized", res_u)
cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image
cv2.imwrite('hw2/output/Q2_equalized_overexposed.png', res_o)
cv2.imwrite('hw2/output/Q2_equalized_underexposed.png', res_u)
plt.hist(equalized_u.ravel(),256,[0,256])
plt.title('Equalized underexposed')
plt.savefig('hw2/output/Q2_underexposed_equalized_hist.png')
plt.close()
plt.hist(equalized_o.ravel(),256,[0,256])
plt.title('Equalized overexposed')
plt.savefig('hw2/output/Q2_overexposed_equalized_hist.png')
plt.close()