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
    # files = get_files_test(folder=os.path.join(root_dir, train_folder),extension_filter=img_extension)
    files = get_files_val(folder=os.path.join(root_dir, val_folder),extension_filter=img_extension)
    print('Num Files: ', len(files) )
    print('Running clustering....')
    i = 0
    num_fail = 0
    for file in tqdm(files):
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
        # bandwidth = cluster.estimate_bandwidth(imgflat,quantile=0.2,n_jobs=-1)
        # bandwidth = 25
        # clt = cluster.MeanShift(bandwidth=bandwidth,bin_seeding=True,max_iter=1000,n_jobs=-1)
        clt = cluster.KMeans(n_clusters=4,max_iter=1000,n_jobs=-1,precompute_distances=True,tol=1e-4)
        # clt = cluster.AgglomerativeClustering(n_clusters=None,distance_threshold=5000,linkage='ward')
        # clt = hdbscan.HDBSCAN(min_cluster_size=50,min_samples=25)
        labels = clt.fit_predict(imgflat)  
        # labels = np.ones((64*64)) 
        # labels[labels == 255] = 0     
        # labels += 1
        # num_uniqueLables = np.unique(labels)
        # print(num_uniqueLables)
        # print(np.max(num_uniqueLables))
        labels = np.reshape(labels,(112,112)).astype(np.uint8)
        labels = Image.fromarray(labels)
        labels = labels.resize((224,224),resample=PIL.Image.NEAREST)
        x = re.split("/home/ken/Documents/Dataset/ImageNet/val/ILSVRC2012_val_(.+).JPEG", file, 1)
        filename = x[1]
        savepath = root_dir + 'kmeans/val/' + filename + '.png'
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

    print('Done clustering....')
    print('num failed: ',num_fail)
