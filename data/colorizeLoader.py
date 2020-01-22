import os
import glob
import torch
import random
# import PIL
import numpy as np
# from . import utils
import torch.utils.data as data
from PIL import Image, ImageOps, ImageCms
from torchvision import transforms
import matplotlib.pyplot as plt
from collections import OrderedDict
import torchvision.transforms.functional as TF

class colorizeLoader(data.Dataset):
    train_folder = "ImageNet/train"
    val_folder = "ImageNet/val"
    img_extension = '.JPEG'
    
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.mode = mode

        if self.mode.lower() == 'train':
            self.train_data = self.get_files(folder=os.path.join(root_dir, self.train_folder),extension_filter=self.img_extension )
        elif self.mode.lower() == 'val':
            self.val_data = self.get_files_val(folder=os.path.join(root_dir, self.val_folder),extension_filter=self.img_extension)
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val")

    def __getitem__(self, index):
        if self.mode.lower() == 'train':
            data_path = self.train_data[index]
        elif self.mode.lower() == 'val':
            data_path = self.val_data[index]
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val")

        # print(data_path)
        img = Image.open(data_path)
        img = img.resize((224,224))

        srgb_p = ImageCms.createProfile("sRGB")
        lab_p  = ImageCms.createProfile("LAB")
        rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
        img = ImageCms.applyTransform(img, rgb2lab)
        
        (L,A,B) = img.split()
        img = transforms.ToTensor()(img)
        L = transforms.ToTensor()(L)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        Lstack = torch.cat((L,L,L),0)
        return Lstack, A, B 
    
    def get_files(self,folder,extension_filter):
        files = []
        folders = os.listdir(folder)
        folders.sort()
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
    train_set = colorizeLoader(root_dir="/home/ken/Documents/Dataset/", mode='val')
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    L, A, B = iter(train_loader).next()
    