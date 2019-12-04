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
import matplotlib.pyplot as plt
from collections import OrderedDict
import torchvision.transforms.functional as TF

class relpatchLoader(data.Dataset):
    #dataset root folders
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


        img = Image.open(data_path)
        img = img.convert('RGB')
        img = img.resize((225,225))
        tiles = []
        for i in range(3):
            for j in range(3):
                x1 = 0 + (75*j) + random.randint(0, 11)
                y1 = 0 + (75*i) + random.randint(0, 11)
                x2 = x1 + 64
                y2 = y1 + 64
                tile = img.crop((x1,y1,x2,y2))
                tile = transforms.ToTensor()(tile)
                tiles.append(tile)

        img = transforms.ToTensor()(img)
        lst = [0,1,2,3,5,6,7,8]
        label = lst[np.random.randint(len(lst))]
        tile1 = tiles[4]
        tile2 = tiles[label]
        return img, tile1, tile2, label
    
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
    train_set = relpatchLoader(root_dir="/home/ken/Documents/Dataset/", mode='val')
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    img, tile1, tile2, label = iter(train_loader).next()
    tile1 = transforms.ToPILImage(mode='RGB')(tile1.squeeze())
    tile2 = transforms.ToPILImage(mode='RGB')(tile2.squeeze())
    print(label)
    plt.subplot(121)
    plt.imshow(tile1)
    plt.subplot(122)
    plt.imshow(tile2)
    plt.show()

    
    