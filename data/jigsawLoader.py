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

class jigsawLoader(data.Dataset):
    #dataset root folders
    train_folder = "ImageNet/train"
    val_folder = "ImageNet/val"
    img_extension = '.JPEG'
    
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.permutations = self.__retrive_permutations(1000)

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
        tiles = torch.stack(tiles, 0)
        order = np.random.randint(len(self.permutations))
        perm = self.permutations[order]
        tiles = tiles[perm]
        return tiles, order
    
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

    def __retrive_permutations(self, classes):
        all_perm = np.load('/home/ken/Documents/Quad-S-Learning/data/permutations_1000.npy')
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

if __name__ == "__main__":
    import utils
    import matplotlib.pyplot as plt
    train_set = jigsawLoader(root_dir="/home/ken/Documents/Dataset/", mode='val')
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    tiles, order = iter(train_loader).next()
    plt.subplot(331)
    plt.imshow(transforms.ToPILImage(mode='RGB')(tiles.squeeze()[0]))
    plt.subplot(332)
    plt.imshow(transforms.ToPILImage(mode='RGB')(tiles.squeeze()[1]))
    plt.subplot(333)
    plt.imshow(transforms.ToPILImage(mode='RGB')(tiles.squeeze()[2]))
    plt.subplot(334)
    plt.imshow(transforms.ToPILImage(mode='RGB')(tiles.squeeze()[3]))
    plt.subplot(335)
    plt.imshow(transforms.ToPILImage(mode='RGB')(tiles.squeeze()[4]))
    plt.subplot(336)
    plt.imshow(transforms.ToPILImage(mode='RGB')(tiles.squeeze()[5]))
    plt.subplot(337)
    plt.imshow(transforms.ToPILImage(mode='RGB')(tiles.squeeze()[6]))
    plt.subplot(338)
    plt.imshow(transforms.ToPILImage(mode='RGB')(tiles.squeeze()[7]))
    plt.subplot(339)
    plt.imshow(transforms.ToPILImage(mode='RGB')(tiles.squeeze()[8]))
    plt.show()
