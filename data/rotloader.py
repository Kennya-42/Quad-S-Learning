import os
import glob
import torch
import random
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps
import numbers
import math
from torchvision import transforms
# import torchvision.transforms.functional as F

class Rotloader(data.Dataset):
    #dataset root folders
    train_folder = "ImageNet/train"
    val_folder = "ImageNet/val"
    img_extension = '.JPEG'
    train_folder_city = 'cluster/city/train'
    val_folder_city = 'cluster/city/val'
    
    def __init__(self, root_dir, mode='train', dataset='imagenet', include_extralabel=False):
        self.root_dir = root_dir
        self.mode = mode
        self.dataset = dataset
        self.include_unrotated = include_extralabel

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
        label = random.randint(0,3)
        rot = 90
        rot = rot*label
        if self.dataset =='city':
            if self.mode == 'train':
                # transX = random.randint(-2, 2)
                # transY = random.randint(-2, 2)
                # img = ImageOps.expand(img, border=(transX,transY,0,0), fill=0)
                # img = transforms.RandomCrop(size=(512,512))(img)
                img = img.resize((224,224),Image.BILINEAR)
            else:
                # img = transforms.CenterCrop(size=(512,512))(img)
                # img = transforms.RandomCrop(size=(512,512))(img)
                img = img.resize((224,224),Image.BILINEAR)
        # if self.mode == 'train':
        #     color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        #     img = transforms.RandomApply([color_jitter], p=0.8)(img)
        img_og = img    
        img = transforms.functional.rotate(img, angle=rot)
        img = transforms.ToTensor()(img)
        # if self.dataset =='imagenet':
        #     img = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(img)
        # elif self.dataset =='city':
        #     img = transforms.Normalize(mean=[0.28689554, 0.32513303, 0.28389177],std=[0.18696375, 0.19017339, 0.18720214])(img)
        #     pass

        if self.include_unrotated:
            img_og = transforms.ToTensor()(img_og)
            return img, label, img_og
        else:
            return img, label
    
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

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for attempt in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                if isinstance(value, numbers.Number):
                    v = value
                elif isinstance(value, torch._six.string_classes):
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                elif isinstance(value, (list, tuple)):
                    v = torch.tensor(value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if random.uniform(0, 1) < self.p:
            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value)
            img[:, x:x + h, y:y + w] = v
        return img

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_set = Rotloader(root_dir="/home/ken/Documents/Dataset/", mode='train', dataset='city')
    print('Dataset Size: ',len(train_set))
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    timages, tlabels = iter(train_loader).next()
    img = transforms.ToPILImage(mode='RGB')(timages[0])
    print(tlabels[0].data.numpy())
    print(img.size)
    plt.imshow(img)
    plt.show()