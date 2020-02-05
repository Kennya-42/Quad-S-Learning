import os
import glob
import torch
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

class fulljigsawLoader(data.Dataset):
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
        groundtruth = img
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
        print('tiles: ',tiles.size())
        black1 = torch.zeros(3,64,10)
        black2 = torch.zeros(3,10,212)
        stack1 = torch.cat([tiles[0],black1,tiles[1],black1,tiles[2]],2)
        stack2 = torch.cat([tiles[3],black1,tiles[4],black1,tiles[5]],2)
        stack3 = torch.cat([tiles[6],black1,tiles[7],black1,tiles[8]],2)
        shuffled = torch.cat([stack1,black2,stack2,black2,stack3],1)
        shuffled = torch.nn.ZeroPad2d((6,7,6,7))(shuffled)
        print('shuffle: ',shuffled.size())

        return shuffled, img
    
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
        print(self.root_dir)
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
    shuffled, img = iter(train_loader).next()
    print(shuffled[0].size())
    img = transforms.ToPILImage(mode='RGB')(img[0])
    shuffled = transforms.ToPILImage(mode='RGB')(shuffled[0])
    
    plt.subplot(211)
    plt.imshow(img)
    plt.subplot(212)
    plt.imshow(shuffled)
    plt.show()
