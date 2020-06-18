import os
import glob
import torch
import random
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps
from torchvision import transforms

class jigsawLoader(data.Dataset):
    #dataset root folders
    train_folder = "ImageNet/train"
    val_folder = "ImageNet/val"
    img_extension = '.JPEG'
    train_folder_city = 'cluster/city/train'
    val_folder_city = 'cluster/city/val'
    
    def __init__(self, root_dir, mode='train', dataset='imagenet'):
        self.root_dir = root_dir
        self.perm_dir = root_dir[:-8] + 'Quad-S-Learning/data/permutations_1000.npy'
        self.mode = mode
        self.permutations = self.__retrive_permutations(1000)
        self.dataset = dataset

        if self.mode.lower() == 'train':
            if self.dataset =='imagenet':
                self.train_data = self.get_files(folder=os.path.join(root_dir, self.train_folder),extension_filter=self.img_extension)#[:50000]
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
        if self.dataset == 'imagenet':
            img = transforms.RandomHorizontalFlip(p=0.5)(img)
            img = img.resize((360,360),Image.BILINEAR)
            img = transforms.CenterCrop(255)(img)
        elif self.dataset == 'city':
            pass
            #resize 256, 255 center crop
            img = transforms.RandomHorizontalFlip(p=0.5)(img)
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)
            img = ImageOps.expand(img, border=(transX,transY,0,0), fill=0)
            img = transforms.RandomCrop(size=(1000,2000))(img)
            img = img.resize((225,225),Image.BILINEAR)
        tiles = []
        for i in range(3):
            for j in range(3):
                x1 = 0 + (75*j) + random.randint(0, 11)
                y1 = 0 + (75*i) + random.randint(0, 11)
                x2 = x1 + 64
                y2 = y1 + 64
                tile = img.crop((x1,y1,x2,y2))
                # tile = transforms.Lambda(rgb_jittering)(tile)
                tile = transforms.ToTensor()(tile)
                if self.dataset =='imagenet':
                    pass
                    # tile = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(tile)
                elif self.dataset =='city':
                    tile = transforms.Normalize(mean=[0.28689554, 0.32513303, 0.28389177],std=[0.18696375, 0.19017339, 0.18720214])(tile)
                # m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
                # tile = transforms.Normalize(mean=m.tolist(), std=s.tolist())(tile)
                tiles.append(tile)

        img = transforms.ToTensor()(img)
        tiles = torch.stack(tiles, 0)
        order = np.random.randint(len(self.permutations))
        perm = self.permutations[order]
        tiles = tiles[perm]
        return tiles, order, img
    
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
        all_perm = np.load(self.perm_dir)
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

def rgb_jittering(im):
    im = np.array(im,np.float32)
    for ch in range(3):
        thisRand = np.random.uniform(0.8, 1.2)
        im[:,:,ch] *= thisRand
    shiftVal = np.random.randint(0,6)
    if np.random.randint(2) == 1:
        shiftVal = -shiftVal
    im += shiftVal
    im = im.astype(np.uint8)
    im = im.astype(np.float32)
    return im

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_set = jigsawLoader(root_dir="/home/ken/Documents/Dataset/", mode='train', dataset='imagenet')
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    tiles, order,img = iter(train_loader).next()
    img = transforms.ToPILImage(mode='RGB')(img.squeeze())
    # img.save('jigsawcityexample.png')
    img.save('jigsawexample.png')
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
