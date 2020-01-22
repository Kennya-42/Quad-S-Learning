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

class Imagenet(data.Dataset):
    #dataset root folders
    train_folder = "ImageNet/train"
    val_folder = "ImageNet/val"
    val_label_txt = "ImageNet/imagenet_val_gt.txt"
    img_extension = '.JPEG'
    
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.mode = mode

        if self.mode.lower() == 'train':
            self.train_data, self.train_labels = self.get_files_and_labels(folder=os.path.join(root_dir, self.train_folder),extension_filter=self.img_extension )
        elif self.mode.lower() == 'val':
            self.val_data,self.val_labels = self.get_files_and_labels_val(folder=os.path.join(root_dir, self.val_folder),extension_filter=self.img_extension,label_path=os.path.join(root_dir, self.val_label_txt))
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val")

    def __getitem__(self, index):
        if self.mode.lower() == 'train':
            data_path = self.train_data[index]
        elif self.mode.lower() == 'val':
            data_path = self.val_data[index]
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val")

        try:
            img = Image.open(data_path)
            img = img.convert('RGB')
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(img)

            if self.mode.lower() == 'train':
                label = self.train_labels[index]
            elif self.mode.lower() == 'val':
                label = self.val_labels[index] 
        except:
            print('FAILED TO LOAD: ', data_path )
            img = torch.rand( 3, 224, 224)
            label = torch.randint(0,999)


        return img, label
    
    def get_files_and_labels(self,folder,extension_filter):
        files = []
        labels = []
        folders = os.listdir(folder)
        folders.sort()
        #index is class label associated with class_folder
        for index, class_folder in enumerate(folders):
            f = glob.glob(os.path.join(folder, class_folder)+'/*'+extension_filter)
            for file in f:
                files.append(file)
                labels.append(index)

        return files,labels

    def get_files_and_labels_val(self,folder,extension_filter,label_path):
        files = glob.glob(folder+'/*'+extension_filter)
        files.sort()
        labels = []
        with open(label_path) as fp:
            line = fp.readline()            
            while line:
                labels.append(int(line))
                line = fp.readline()
        return files,labels

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
    train_set = Imagenet(root_dir="/home/ken/Documents/Dataset/", mode='val')
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    timages, tlabels = iter(train_loader).next()
    img = transforms.ToPILImage(mode='RGB')(timages[0])
    print(tlabels[0].data.numpy())
    plt.imshow(img)
    plt.show()