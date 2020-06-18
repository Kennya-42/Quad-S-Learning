import os
from PIL import Image, ImageOps, ImageCms
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
import glob
import cv2
import math
img = Image.open("Raw_otsu.png")
img = img.convert('L')
img = np.array(img)
thresholds = threshold_multiotsu(img,classes=4)
print(thresholds)
print(img.min(),img.max())
plt.hist(img.ravel(), bins = 256, color = 'Blue')
plt.axvline(x=thresholds[0], color='r', linestyle='solid', linewidth=1)
plt.axvline(x=thresholds[1], color='r', linestyle='solid', linewidth=1)
plt.axvline(x=thresholds[2], color='r', linestyle='solid', linewidth=1)
plt.xlabel('Pixel Intensity')
plt.title('Histogram')
plt.ylabel('Count')
plt.show()