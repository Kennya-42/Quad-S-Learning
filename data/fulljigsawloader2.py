import os
import glob
import torch
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

class fulljigsawLoader2(data.Dataset):
    #dataset root folders
    train_folder = "ImageNet/train"
    val_folder = "ImageNet/val"
    img_extension = '.JPEG'
    train_folder_city = 'cluster/city/train'
    val_folder_city = 'cluster/city/val'
    
    def __init__(self, root_dir, mode='train', dataset='city',include_extralabel=False):
        self.root_dir = root_dir
        self.perm_dir = root_dir[:-8] + 'Quad-S-Learning/data/permutations_1000.npy'
        self.mode = mode
        self.permutations = self.__retrive_permutations()
        self.dataset = dataset
        self.include_extralabel = include_extralabel

        if self.mode.lower() == 'train':
            if self.dataset =='imagenet':
                self.train_data = self.get_files(folder=os.path.join(root_dir, self.train_folder),extension_filter=self.img_extension)
            elif self.dataset =='city':
                self.train_data = self.get_files_val(folder=os.path.join(root_dir, self.train_folder_city),extension_filter='.png')
        elif self.mode.lower() == 'val':
            if self.dataset =='imagenet':
                self.val_data = self.get_files_val(folder=os.path.join(root_dir, self.val_folder),extension_filter=self.img_extension)
            elif self.dataset =='city':
                self.val_data = self.get_files_val(folder=os.path.join(root_dir, self.val_folder_city),extension_filter='.png')
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
        img = img.resize((960,960))
        gt = transforms.ToTensor()(img)
        tiles = []
        for i in range(3):
            for j in range(3):
                x1 = 0 + (320*i)
                y1 = 0 + (320*j)
                x2 = x1 + 320
                y2 = y1 + 320
                tile = img.crop((x1,y1,x2,y2))
                # tile = transforms.RandomCrop(size=(512,512))(tile)
                tile = transforms.ToTensor()(tile)
                tiles.append(tile)
        tiles = torch.stack(tiles, 0)
        order = np.random.randint(len(self.permutations))
        perm = self.permutations[order]
        tiles = tiles[perm]
        stack1 = torch.cat([tiles[0], tiles[1], tiles[2]], 2)
        stack2 = torch.cat([tiles[3], tiles[4], tiles[5]], 2)
        stack3 = torch.cat([tiles[6], tiles[7], tiles[8]], 2)
        shuffled = torch.cat([stack1,stack2,stack3],1)
        return shuffled, order, gt
        
    
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

    def __retrive_permutations(self):
        all_perm = np.load(self.perm_dir)
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

if __name__ == "__main__":
    import utils
    import matplotlib.pyplot as plt
    train_set = fulljigsawLoader2(root_dir="/home/ken/Documents/Dataset/", mode='train')
    print(len(train_set))
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    shuffled, order, gt = iter(train_loader).next()
    gt = transforms.ToPILImage(mode='RGB')(gt[0])
    shuffled = transforms.ToPILImage(mode='RGB')(shuffled[0])
    
    plt.subplot(211)
    plt.imshow(gt)
    plt.subplot(212)
    plt.imshow(shuffled)
    plt.show()
