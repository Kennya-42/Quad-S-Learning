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
# train_folder = "Cityscapes/leftImg8bit/train"
# train_lbl_folder = "Cityscapes/gtFine/train"
# val_folder = "Cityscapes/leftImg8bit/val"
# val_lbl_folder = "Cityscapes/gtFine/val"
test_folder = 'cluster/city/val'
img_extension = '.png'
lbl_name_filter = 'labelTrainIds'
root_dir="/home/ken/Documents/Dataset/"

def get_files_train_city(folder, name_filter=None, extension_filter=None):
    
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))
    
    if name_filter is None:
        name_cond = lambda filename: True
    else:
        name_cond = lambda filename: name_filter in filename

    if extension_filter is None:
        ext_cond = lambda filename: True
    else:
        ext_cond = lambda filename: filename.endswith(extension_filter)

    filtered_files = []
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_cond(file) and ext_cond(file):
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)
    
    return filtered_files

def get_files_train(folder,extension_filter):
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
    # files = get_files_train_city(folder=os.path.join(root_dir, train_folder),extension_filter=img_extension)
    files = get_files_val(folder=os.path.join(root_dir, test_folder),extension_filter=img_extension)
    print('Num Files: ', len(files) )
    print('Running otsu....')
    for file in tqdm(files):
        # print(file)
        img_original = Image.open(file).convert('RGB')
        img = img_original
        img = img.convert('L')
        img = img.resize((512,256),resample=PIL.Image.BILINEAR)
        w,h = img.size
        print(h,w)
        img = np.array(img)
        # imgflat = np.reshape(img,(-1,1))
        # print(imgflat.shape)
        # imgflat = imgflat - imgflat.mean()
        start = time.time()
        thresholds = threshold_multiotsu(img,classes=8)
        end = time.time() - start
        print(end)
        labels = np.digitize(img, bins=thresholds)
        # print(labels.shape)
        
        # labels = np.reshape(labels,(256,512)).astype(np.uint8)
        labels = Image.fromarray(labels.astype(np.uint8)).convert('L')
        labels = labels.resize((2048,1024),resample=PIL.Image.NEAREST)
        # print(labels.size)
        x = re.split("/home/ken/Documents/Dataset/cluster/city/val/(.+)", file, 1)
        filename = x[1]
        savepath = root_dir + 'cluster/otsu/val_city_20c/' + filename
        print(savepath)
        # labels.save(savepath)
        
        plt.subplot(121)
        plt.imshow(img_original)
        plt.subplot(122)
        plt.imshow(labels)
        plt.show()
        exit()
        

    print('Finishing....')
