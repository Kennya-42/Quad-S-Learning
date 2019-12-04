import torchvision.transforms.functional as TF
from . import custom_transforms as tr
from . import utils
from torchvision import transforms
from collections import OrderedDict
import torch.utils.data as data
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import os
# import PIL
# from . import augment
# from . import autoaugment

def pil_loader(data_path, label_path):
    data = Image.open(data_path)
    label = Image.open(label_path)
    return data, label

class Cityscapes(data.Dataset):
    #dataset root folders
    train_folder = "Cityscapes/leftImg8bit/train"
    train_lbl_folder = "Cityscapes/gtFine/train"
    val_folder = "Cityscapes/leftImg8bit/val"
    val_lbl_folder = "Cityscapes/gtFine/val"
    test_folder = val_folder
    test_lbl_folder = val_lbl_folder
    img_extension = '.png'
    lbl_name_filter = 'labelTrainIds'
    color_encoding = OrderedDict([
            ('road', (128, 64, 128)),
            ('sidewalk', (244, 35, 232)),
            ('building', (70, 70, 70)),
            ('wall', (102, 102, 156)),
            ('fence', (190, 153, 153)),
            ('pole', (153, 153, 153)),
            ('traffic_light', (250, 170, 30)),
            ('traffic_sign', (220, 220, 0)),
            ('vegetation', (107, 142, 35)),
            ('terrain', (152, 251, 152)),
            ('sky', (70, 130, 180)),
            ('person', (220, 20, 60)),
            ('rider', (255, 0, 0)),
            ('car', (0, 0, 142)),
            ('truck', (0, 0, 70)),
            ('bus', (0, 60, 100)),
            ('train', (0, 80, 100)),
            ('motorcycle', (0, 0, 230)),
            ('bicycle', (119, 11, 32)),
            ('unlabeled', (0, 0, 0))
    ])

    def __init__(self, root_dir, mode='train', loader=pil_loader,height=1024, width=2048):
        self.root_dir = root_dir
        self.mode = mode
        self.loader = loader
        self.height = height
        self.width = width

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = utils.get_files(os.path.join(root_dir, self.train_folder), extension_filter=self.img_extension)
            self.train_labels = utils.get_files(os.path.join(root_dir, self.train_lbl_folder),name_filter=self.lbl_name_filter,extension_filter=self.img_extension)
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = utils.get_files( os.path.join(root_dir, self.val_folder), extension_filter=self.img_extension)
            self.val_labels = utils.get_files(os.path.join(root_dir, self.val_lbl_folder), name_filter=self.lbl_name_filter, extension_filter=self.img_extension)
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = utils.get_files(os.path.join(root_dir, self.test_folder),extension_filter=self.img_extension)
            self.test_labels = utils.get_files(os.path.join(root_dir, self.test_lbl_folder),name_filter=self.lbl_name_filter,extension_filter=self.img_extension)
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test")

    def __getitem__(self, index):
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[index]
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)
        img = img.convert('RGB')
        label = label.convert('P')
        
        # mask = np.array(label).astype(np.int64)
        # mask = torch.from_numpy(mask)
        # mask = utils.LongTensorToRGBPIL(None)(mask)
        # f, axarr = plt.subplots(2,2)
        # axarr[0,0].imshow(img)
        # axarr[0,1].imshow(mask)
        # Mixup https://forums.fast.ai/t/mixup-data-augmentation/22764
        # PIL.Image.blend(im1, im2, alpha) #interpolate 2 images
        # Scale hue, saturation, and brightness with coefficientsuniformly drawn from [0.6,1.4]
        # add PCA noise from normal dist N(0,0.1)
        sample = {'image': img, 'label': label}

        if self.mode.lower() == 'train':
            sample = self.transform_tr(sample)
        else:
            sample = self.transform_val(sample)

        img , label = sample['image'],sample['label']
        
        return img, label

    
    def transform_tr(self,input):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomCrop(self.width, self.height),
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
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test")

if __name__ == "__main__":
    import utils
    import custom_transforms as tr
    train_set = Cityscapes(root_dir="/home/ken/Documents/Dataset/", mode='train',height=512, width=1024)
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    timages, tlabels = iter(train_loader).next()
    plt.imshow(tlabels.squeeze())
    plt.show()
    img = transforms.ToPILImage(mode='RGB')(timages[0])
    label = utils.LongTensorToRGBPIL(None)(tlabels.squeeze())
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img)
    axarr[1].imshow(label)
    plt.show()