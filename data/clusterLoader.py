import os
import glob
import torch
import random
import PIL
import numpy as np
# from . import utils
import torch.utils.data as data
from PIL import Image, ImageOps
from torchvision import transforms

from collections import OrderedDict
import torchvision.transforms.functional as TF

class clusterLoader(data.Dataset):
    #dataset root folders
    train_folder = "ImageNet/train"
    train_folder_gt = "kmeans/train"
    val_folder = "ImageNet/val"
    val_folder_gt = "kmeans/val"
    img_extension = '.JPEG'
    label_extension = '.png'
    
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.mode = mode

        if self.mode.lower() == 'train':
            self.train_data = self.get_files_train(folder=os.path.join(root_dir, self.train_folder),extension_filter=self.img_extension )
            self.train_data_gt = self.get_files_val(folder=os.path.join(root_dir, self.train_folder_gt),extension_filter=self.label_extension )
        elif self.mode.lower() == 'val':
            self.val_data = self.get_files_val(folder=os.path.join(root_dir, self.val_folder),extension_filter=self.img_extension)
            self.val_data_gt = self.get_files_val(folder=os.path.join(root_dir, self.val_folder_gt),extension_filter=self.label_extension )
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val")

    def __getitem__(self, index):
        if self.mode.lower() == 'train':
            data_path = self.train_data[index]
            label_path = self.train_data_gt[index]
        elif self.mode.lower() == 'val':
            data_path = self.val_data[index]
            label_path = self.val_data_gt[index]
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val")

        img = Image.open(data_path)
        label = Image.open(label_path)
        img = img.convert('RGB')
        label = label.convert('L')
        img = transforms.ToTensor()(img)
        label = transforms.ToTensor()(label)
        label = label.type(torch.long).squeeze()
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(img)
        # label = 1
        return img, label
    
    def get_files_train(self,folder,extension_filter):
        files = []
        folders = os.listdir(folder)
        folders.sort()
        #index is class label associated with class_folder
        for index, class_folder in enumerate(folders):
            f = glob.glob(os.path.join(folder, class_folder)+'/*'+extension_filter)
            for file in f:
                files.append(file)

        return files

    def get_files_val(self,folder,extension_filter):
        files = glob.glob(folder+'/*'+extension_filter)
        files.sort()
        return files

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test")

if __name__ == "__main__":
    import utils
    import matplotlib.pyplot as plt
    train_set = clusterLoader(root_dir="/home/ken/Documents/Dataset/", mode='val')
    train_loader = data.DataLoader(train_set, batch_size=2, shuffle=False, num_workers=0)
    img, label = iter(train_loader).next()
    img = transforms.ToPILImage(mode='L')(label[0])
    plt.imshow(img)
    plt.show()
    