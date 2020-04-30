import cv2
import numpy as np
from matplotlib import pyplot as plt

# Open the image.
underexposed = cv2.imread("hw2/input/underexposed.jpg")
overexposed = cv2.imread("hw2/input/overexposed.jpg")
underexposed = cv2.resize(underexposed, (1000, 750))
overexposed = cv2.resize(overexposed, (1000, 750))
gray_underexposed = cv2.cvtColor(underexposed, cv2.COLOR_BGR2GRAY)
gray_overexposed = cv2.cvtColor(overexposed, cv2.COLOR_BGR2GRAY)

# Trying 4 gamma values.
for gamma in [0.1, 0.5, 1.0, 1.2, 1.7, 2.2]:
    # Apply gamma correction.
    gamma_corrected_underexposed = np.array(
        255 * (gray_underexposed / 255) ** gamma, dtype="uint8"
    )
    gamma_corrected_overexposed = np.array(
        255 * (gray_overexposed / 255) ** gamma, dtype="uint8"
    )
    cv2.imshow("gamma: " + str(gamma) + " underexposed", gamma_corrected_underexposed)
    cv2.imshow("gamma: " + str(gamma) + " overexposed", gamma_corrected_overexposed)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image

# 0.5 looks best for the underexposed, 1.7 looks best for the overexposed

final_gcu = np.array(255 * (gray_underexposed / 255) ** 0.5, dtype="uint8")
final_gco = np.array(255 * (gray_overexposed / 255) ** 1.7, dtype="uint8")

plt.hist(gray_underexposed.ravel(),256,[0,256])
plt.title('Original underexposed')
plt.savefig('hw2/output/Q1_original_underexposed_hist.png')
plt.close()
plt.hist(gray_overexposed.ravel(),256,[0,256])
plt.title('Original overexposed')
plt.savefig('hw2/output/Q1_original_overexposed_hist.png')
plt.close()
plt.hist(final_gcu.ravel(),256,[0,256])
plt.title('Gamma corrected 0.5 underexposed')
plt.savefig('hw2/output/Q1_gamma_underexposed_hist.png')
plt.close()
plt.hist(final_gco.ravel(),256,[0,256])
plt.title('Gamma corrected 1.7 overexposed')
plt.savefig('hw2/output/Q1_gamma_overexposed_hist.png')
plt.close()

res_o = np.hstack((gray_overexposed, final_gco))
res_u = np.hstack((gray_underexposed, final_gcu))
cv2.imwrite('hw2/output/Q1_gamma_overexposed.png', res_o)
cv2.imwrite('hw2/output/Q1_gamma_underexposed.png', res_u)