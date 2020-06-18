import os
import glob
import torch
import numpy as np
import random
import torch.utils.data as data
from PIL import Image, ImageOps, ImageCms
from torchvision import transforms
from skimage.color import lab2rgb, rgb2lab

class colorizeLoader(data.Dataset):
    train_folder = "ImageNet/train"
    val_folder = "ImageNet/val"
    img_extension = '.JPEG'
    train_folder_city = 'cluster/city/train'
    val_folder_city = 'cluster/city/val'
    
    def __init__(self, root_dir, mode='train', dataset='imagenet'):
        self.root_dir = root_dir
        self.mode = mode
        self.dataset = dataset

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
        
        img = Image.open(data_path).convert('RGB')
        img = transforms.RandomHorizontalFlip(p=0.5)(img)
        transX = random.randint(-2, 2)
        transY = random.randint(-2, 2)
        img = ImageOps.expand(img, border=(transX,transY,0,0), fill=0)
        if self.dataset =='city':
            img = transforms.RandomCrop(size=(512,512))(img)
        img = img.resize((224,224))
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        img = transforms.RandomApply([color_jitter], p=0.8)(img)
       
        img = np.asarray(img)
        img_og = img
        img = img/255.0 #normalize to 0 to 1.
        img_lab = rgb2lab(img) #[0,100] x [-128,128] x [-128,128] LAB ranges
        img_ab = img_lab[:, :, 1:3]
        img_ab =img_ab+128
        img_ab = np.moveaxis(img_ab,2,0)
        img_ab = torch.from_numpy(img_ab)
        img_gray = img_lab[:,:,0]
        img_gray = torch.from_numpy(img_gray).unsqueeze(0)
        img_gray = torch.cat((img_gray,img_gray,img_gray),0)
        return img_gray.float(), img_ab.float(), img_og
    
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
    import matplotlib.pyplot as plt
    train_set = colorizeLoader(root_dir="/home/ken/Documents/Dataset/", mode='train',dataset='city')
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    print(len(train_set))
    img_gray, img_ab, img = iter(train_loader).next()
    print(img_gray.min(),img_gray.max())
    print(img_ab.min(),img_ab.max())
    # exit()
    img_gray, img_ab,img = img_gray.squeeze(),img_ab.squeeze(),img.squeeze()
    img_lab = torch.zeros(3,224,224)
    img_lab[0,:,:] = img_gray[0,:,:]
    img_lab[1:3,:,:] = img_ab - 128.0
    img_lab = img_lab.numpy()
    img_lab = np.moveaxis(img_lab,0,-1)
    img_rgb = lab2rgb(img_lab)*255
    img_rgb = img_rgb.astype(np.uint8)

    
    plt.subplot(311)
    plt.imshow(img_gray[0],cmap='gray')
    plt.subplot(312)
    plt.imshow(img)
    plt.subplot(313)
    plt.imshow(img_rgb)
    plt.show()
    