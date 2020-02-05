import os
from PIL import Image, ImageOps, ImageCms
import PIL
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from sklearn.cluster import MeanShift,KMeans
from skimage.filters import threshold_multiotsu
from sklearn import mixture
from sklearn import cluster
from tqdm import tqdm
import time
import re
import hdbscan

train_folder = "ImageNet/train"
val_folder = "ImageNet/val"
img_extension = '.JPEG'
root_dir="/home/ken/Documents/Dataset/"

def get_files_test(folder,extension_filter):
    files = []
    folders = os.listdir(folder)
    folders.sort()
    #index is class label associated with class_folder
    for index, class_folder in enumerate(folders):
        f = glob.glob(os.path.join(folder, class_folder)+'/*'+extension_filter)
        for file in f:
            files.append(file)

    return files

def get_files_val(folder,extension_filter):
    files = glob.glob(folder+'/*'+extension_filter)
    files.sort()
    return files

if __name__ == "__main__":
    files = get_files_test(folder=os.path.join(root_dir, train_folder),extension_filter=img_extension)
    # files = get_files_val(folder=os.path.join(root_dir, val_folder),extension_filter=img_extension)
    print('Num Files: ', len(files) )
    print('Running otsu....')
    i = 0
    num_fail = 0
    for file in tqdm(files):
        # print(file)
        img_original = Image.open(file).convert('RGB')
        img = img_original.resize((112,112),resample=PIL.Image.BICUBIC)
        img = np.array(img)

        try:
            imgflat = np.reshape(img,(-1,3))
        except:
            print(file)
            print(img.shape)
            exit()
        imgflat = imgflat - imgflat.mean()
        
        thresholds = threshold_multiotsu(imgflat,classes=4)
        labels = np.digitize(imgflat, bins=thresholds)
        # print(labels.shape)
        
        # num_uniqueLables = np.unique(labels)
        # print(num_uniqueLables)
        # print(np.max(num_uniqueLables))
        labels = np.reshape(labels,(112,112,3)).astype(np.uint8)
        labels = Image.fromarray(labels).convert('L')
        labels = labels.resize((224,224),resample=PIL.Image.NEAREST)
        # print(labels.size)
        x = re.split("/home/ken/Documents/Dataset/ImageNet/train/n\d+/(.+).JPEG", file, 1)
        filename = x[1]
        savepath = root_dir + 'otsu/train/' + filename + '.png'
        # print(savepath)
        labels.save(savepath)
        
        # if num_uniqueLables <=1 or num_uniqueLables > 10:
        #     num_fail += 1

        # plt.subplot(121)
        # plt.imshow(img_original)
        # plt.subplot(122)
        # plt.imshow(labels)
        # plt.show()
        # if i >0:
        #     break
        # i += 1

    print('Finishing....')
    print('num failed: ',num_fail)
