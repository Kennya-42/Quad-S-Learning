import os
from PIL import Image, ImageOps, ImageCms
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

srgb_p = ImageCms.createProfile("sRGB")
lab_p  = ImageCms.createProfile("LAB")
rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, "LAB",  "RGB")


img = Image.open("ILSVRC2012_val_00000001.JPEG")
img = ImageCms.applyTransform(img, rgb2lab)
# img = ImageCms.applyTransform(img, lab2rgb)


img = np.array(img)
print(img.min(),img.max())
plt.imshow(img)
plt.show()