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
import cv2
# import torchvision.transforms.functional as F

class simclrloader(data.Dataset):
    #dataset root folders
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

        img = Image.open(data_path)
        img = img.convert('RGB')
        if self.dataset == "city":
            img = img.resize((1024,512),Image.BILINEAR)
            img_og = transforms.ToTensor()(img)
        elif self.dataset == "imagenet":
            img = img.resize((256,256),Image.BILINEAR)
        xis, xjs = img, img
        xis, xjs = self.data_augment(xis), self.data_augment(xjs)
        return xis, xjs, img_og

    def data_augment(self, img):
        w,h = 224,224
        kernal_size = int(0.1 * h)
        #kernal should be a odd number
        if kernal_size%2 == 0:
            kernal_size += 1
        
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=(h,w)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=kernal_size),
                                              transforms.ToTensor()])

        img = data_transforms(img)
        return img
    
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

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()
        prob = 0

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_set = simclrloader(root_dir="/home/ken/Documents/Dataset/", mode='train', dataset='city')
    print('Dataset Size: ',len(train_set))
    train_loader = data.DataLoader(train_set, batch_size=2, shuffle=False, num_workers=0)
    xis, xjs, img_og = iter(train_loader).next()
    xis = transforms.ToPILImage(mode='RGB')(xis[1])
    xjs = transforms.ToPILImage(mode='RGB')(xjs[1])
    img_og = transforms.ToPILImage(mode='RGB')(img_og[1])
    xis.save('simclr2_transform1.png')
    xjs.save('simclr2_transform2.png')
    img_og.save('simclr2_original.png')
    print(xis.size)
    plt.subplot(311)
    plt.imshow(img_og)
    plt.subplot(312)
    plt.imshow(xis)
    plt.subplot(313)
    plt.imshow(xjs)
    plt.show()