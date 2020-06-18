if __name__ != "__main__":
    from . import custom_transforms as tr
import torchvision.transforms.functional as TF
from torchvision import transforms
from collections import OrderedDict
import torch.utils.data as data
from PIL import Image, ImageOps
import numpy as np
import random
import glob
import torch
import os

def pil_loader(data_path, label_path):
    data = Image.open(data_path)
    label = Image.open(label_path)
    return data, label

class Camvid(data.Dataset):
    #dataset root folders
    train_folder = "camvid/train"
    train_lbl_folder = "camvid/trainannot"
    val_folder = "camvid/val"
    val_lbl_folder = "camvid/valannot"
    label_extension = '.png'
    img_extension = '.png'
   
    color_encoding = OrderedDict([
        ('sky', (128, 128, 128)),
        ('building', (128, 0, 0)),
        ('pole', (192, 192, 128)),
        # ('road_marking', (255, 69, 0)),
        ('road', (128, 64, 128)),
        ('pavement', (60, 40, 222)),
        ('tree', (128, 128, 0)),
        ('sign_symbol', (192, 128, 128)),
        ('fence', (64, 64, 128)),
        ('car', (64, 0, 128)),
        ('pedestrian', (64, 64, 0)),
        ('bicyclist', (0, 128, 192)),
        ('unlabeled', (0, 0, 0))
    ])

    def __init__(self, root_dir, mode='train', loader=pil_loader,height=360, width=480, num_train=-1):
        self.root_dir = root_dir
        self.mode = mode
        self.loader = loader
        self.height = height
        self.width = width
        self.num_train = num_train

        if self.mode.lower() == 'train':
            self.train_data = self.get_files(os.path.join(root_dir, self.train_folder), extension_filter=self.img_extension)
            self.train_labels = self.get_files(os.path.join(root_dir, self.train_lbl_folder),extension_filter=self.img_extension)
        elif self.mode.lower() == 'val':
            self.val_data = self.get_files( os.path.join(root_dir, self.val_folder), extension_filter=self.img_extension)
            self.val_labels = self.get_files(os.path.join(root_dir, self.val_lbl_folder), extension_filter=self.img_extension)
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val")

        if self.num_train > 0:
            self.train_data = self.train_data[:self.num_train]
            self.train_labels = self.train_labels[:self.num_train]

    def __getitem__(self, index):
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[index]
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val")

        img = Image.open(data_path)
        label = Image.open(label_path)
        img = img.convert('RGB')
        sample = {'image': img, 'label': label}

        # if self.mode.lower() == 'train':
        #     sample = self.transform_tr(sample)
        # else:
        #     sample = self.transform_val(sample)
        
        img , label = sample['image'],sample['label']
        
        img = transforms.ToTensor()(img)
        label = np.array(label).astype(np.int64)
        label[label==255] = 11
        label = torch.from_numpy(label)
        # label = label.long().squeeze()
        return img, label

    def get_files(self,folder,extension_filter):
        files = glob.glob(folder+'/*'+extension_filter)
        files.sort()
        return files

    def transform_tr(self,input):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            # tr.RandomCrop(self.width, self.height),
            tr.Resize((self.width, self.height)),
            tr.RandomTranslation(),
            tr.ToTensor()
        ])

        return composed_transforms(input)

    def transform_val(self,input):
        composed_transforms = transforms.Compose([
            tr.Resize((self.width, self.height)),
            tr.ToTensor()
        ])

        return composed_transforms(input)

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val")

if __name__ == "__main__":
    import utils
    import custom_transforms as tr
    import matplotlib.pyplot as plt
    train_set = Camvid(root_dir="/home/ken/Documents/Dataset/", mode='train',height=360, width=480,num_train=-1)
    train_loader = data.DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)
    print(len(train_loader))
    timages, tlabels = iter(train_loader).next()
    print(timages.size())
    print(tlabels.size())
    label = tlabels[0]
    print(label.size())
    print(torch.unique(tlabels[0]))
    img = transforms.ToPILImage(mode='RGB')(timages[0])
    # label = transforms.ToPILImage(mode='L')(tlabels[0])
    plt.subplot(211)
    plt.imshow(img)
    plt.subplot(212)
    plt.imshow(label)
    plt.show()