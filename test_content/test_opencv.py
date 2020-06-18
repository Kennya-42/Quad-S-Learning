import os
from PIL import Image, ImageOps, ImageCms
import PIL
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from sklearn.cluster import MeanShift,KMeans
from sklearn import mixture
from sklearn import cluster
from tqdm import tqdm
import time
import re
import hdbscan

train_folder = "Cityscapes/leftImg8bit/train"
train_lbl_folder = "Cityscapes/gtFine/train"
test_folder = 'cluster/city/val'
val_folder = "Cityscapes/leftImg8bit/val"
val_lbl_folder = "Cityscapes/gtFine/val"
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
    # files = get_files_train_city(folder=os.path.join(root_dir, train_folder),extension_filter=img_extension)
    files = get_files_val(folder=os.path.join(root_dir, test_folder),extension_filter=img_extension)
    print('Num Files: ', len(files) )
    
    print('Running clustering....')
    for i, file in enumerate(tqdm(files)):
        # if i < 18348:
        #     continue
        img_original = Image.open(file).convert('RGB')
        # img = img_original
        # print(file)
        img = img_original.resize((512,1024),resample=PIL.Image.BICUBIC)
        img = np.array(img)
        # 
        try:
            imgflat = np.reshape(img,(-1,3))
        except:
            print(file)
            print(img.shape)
            exit()
        imgflat = imgflat - imgflat.mean()
        clt = cluster.KMeans(n_clusters=10,max_iter=500,n_init=20,n_jobs=-1,precompute_distances=True,tol=1e-4)
        labels = clt.fit_predict(imgflat)  
        labels2 = clt.fit_predict(imgflat)  
        # labels[labels == 255] = 0     
        # labels += 1
        # num_uniqueLables = np.unique(labels)
        # print(num_uniqueLables)
        # print(np.max(num_uniqueLables))
        labels = np.reshape(labels,(1024,512)).astype(np.uint8)
        labels = Image.fromarray(labels)
        labels = labels.resize((2048,1024),resample=PIL.Image.NEAREST)
        labels2 = np.reshape(labels2,(1024,512)).astype(np.uint8)
        labels2 = Image.fromarray(labels2)
        labels2 = labels2.resize((2048,1024),resample=PIL.Image.NEAREST)
        x = re.split("/home/ken/Documents/Dataset/cluster/city/val/(.+)", file, 1)
        filename = x[1]
        savepath = root_dir + 'cluster/kmeans/extra_city/' + filename
        # print(savepath)
        # labels.save(savepath)

        plt.subplot(311)
        plt.imshow(img_original)
        plt.subplot(312)
        plt.imshow(labels)
        plt.subplot(313)
        plt.imshow(labels2)
        plt.show()
        exit()
        

    print('Done clustering....')
