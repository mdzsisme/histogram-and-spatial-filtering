import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram_equalization(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    return equalized_image

def spatial_smoothing(image, kernel_size=(5, 5)):
    smoothed_image = cv2.GaussianBlur(image, kernel_size, 0)
    return smoothed_image

def spatial_sharpening(image):
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

image_path = 'image.jpg'
image = cv2.imread(image_path)  

equalized_image = histogram_equalization(image)
smoothed_image = spatial_smoothing(image)
sharpened_image = spatial_sharpening(image)

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.hist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).ravel(), 256, [0,256])
plt.title('Histogram of Original Image')

plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Image After Histogram Equalization')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.hist(equalized_image.ravel(), 256, [0,256])
plt.title('Histogram of Image After Histogram Equalization')

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(smoothed_image)
plt.title('Image After Spatial Smoothing')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sharpened_image)
plt.title('Image After Spatial Sharpening')
plt.axis('off')

plt.show()
