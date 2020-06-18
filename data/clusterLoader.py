import os
import glob
import torch
import random
import PIL
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps
from torchvision import transforms
if __name__ != "__main__":
    from . import custom_transforms as tr

class clusterLoader(data.Dataset):
    #dataset root folders
    train_folder = "ImageNet/train"
    train_folder_gt_kmeans = "cluster/kmeans/train"
    train_folder_gt_otsu = "cluster/otsu/train"
    train_folder_city = "cluster/city/train"
    train_folder_gt_kmeans_city = "cluster/kmeans/train_city"
    train_folder_gt_otsu_city = "cluster/otsu/train_city"
    val_folder = "ImageNet/val"
    val_folder_gt_kmeans = "cluster/kmeans/val"
    val_folder_gt_otsu = "cluster/otsu/val"
    val_folder_city = "cluster/city/val"
    val_folder_gt_kmeans_city = "cluster/kmeans/val_city"
    val_folder_gt_otsu_city = "cluster/otsu/val_city"
    img_extension = '.JPEG'
    label_extension = '.png'
    
    def __init__(self, root_dir, mode='train', dataset='imagenet', label='kmeans'):
        self.root_dir = root_dir
        self.mode = mode
        self.dataset = dataset
        self.label_type = label

        if self.mode.lower() == 'train':
            if self.dataset == 'imagenet':
                self.train_data = self.get_files_train(folder=os.path.join(root_dir, self.train_folder),extension_filter=self.img_extension )
                if self.label_type == 'kmeans':
                    self.train_data_gt = self.get_files_val(folder=os.path.join(root_dir, self.train_folder_gt_kmeans),extension_filter=self.label_extension )
                elif self.label_type == 'otsu':
                    self.train_data_gt = self.get_files_val(folder=os.path.join(root_dir, self.train_folder_gt_otsu),extension_filter=self.label_extension )
            elif self.dataset == 'city':
                self.train_data = self.get_files_val(folder=os.path.join(root_dir, self.train_folder_city),extension_filter=self.label_extension )
                if self.label_type == 'kmeans':
                    self.train_data_gt = self.get_files_val(folder=os.path.join(root_dir, self.train_folder_gt_kmeans_city),extension_filter=self.label_extension )
                elif self.label_type == 'otsu':
                    self.train_data_gt = self.get_files_val(folder=os.path.join(root_dir, self.train_folder_gt_otsu_city),extension_filter=self.label_extension )
        elif self.mode.lower() == 'val':
            if self.dataset == 'imagenet':
                self.val_data = self.get_files_val(folder=os.path.join(root_dir, self.val_folder),extension_filter=self.img_extension)
                if self.label_type == 'kmeans':
                    self.val_data_gt = self.get_files_val(folder=os.path.join(root_dir, self.val_folder_gt_kmeans),extension_filter=self.label_extension )
                elif self.label_type == 'otsu':
                    self.val_data_gt = self.get_files_val(folder=os.path.join(root_dir, self.val_folder_gt_otsu),extension_filter=self.label_extension )
            elif self.dataset == 'city':
                self.val_data = self.get_files_val(folder=os.path.join(root_dir, self.val_folder_city),extension_filter=self.label_extension)
                if self.label_type == 'kmeans':
                    self.val_data_gt = self.get_files_val(folder=os.path.join(root_dir, self.val_folder_gt_kmeans_city),extension_filter=self.label_extension )
                elif self.label_type == 'otsu':
                    self.val_data_gt = self.get_files_val(folder=os.path.join(root_dir, self.val_folder_gt_otsu_city),extension_filter=self.label_extension )
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val")

    def __getitem__(self, index):
        if self.mode.lower() == 'train':
            data_path = self.train_data[index]
            label_path = self.train_data_gt[index]
        elif self.mode.lower() == 'val':
            data_path = self.val_data[index]
            label_path = self.val_data_gt[index]
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val")
            
        img = Image.open(data_path)
        label = Image.open(label_path)
        img = img.convert('RGB')
        # print(np.max(img),np.min(img))
        label = label.convert('L')
        sample = {'image': img, 'label': label}
        if self.mode == 'train':
            if self.dataset == 'city':
                sample = self.transform_tr(sample)
            else:
                sample = self.transform_tr(sample,224,224)
        else:
            if self.dataset == 'city':
                sample = self.transform_val(sample)
            else:
                sample = self.transform_val(sample,224,224)

        img , label = sample['image'],sample['label']
        # color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        # img = transforms.RandomApply([color_jitter], p=0.8)(img)
        img = transforms.ToTensor()(img)
        img_un = img
        label = np.array(label).astype(np.int64)
        label[label==255] = 0
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)
        if self.dataset == 'imagenet':
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(img)
        return img, label, img_un

    def transform_tr(self,input,h=512,w=1024):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomCrop(w,h),
            tr.RandomTranslation()
        ])

        return composed_transforms(input)

    def transform_val(self,input,h=512,w=1024):
        composed_transforms = transforms.Compose([
            tr.Resize((w,h))
        ])

        return composed_transforms(input)

    def get_files_train(self,folder,extension_filter):
        files = []
        folders = os.listdir(folder)
        folders.sort()
        #index is class label associated with class_folder
        for index, class_folder in enumerate(folders):
            f = glob.glob(os.path.join(folder, class_folder)+'/*'+extension_filter)
            for file in f:
                files.append(file)
        files.sort()
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
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train and val")

if __name__ == "__main__":
    import utils
    import matplotlib.pyplot as plt
    import custom_transforms as tr
    train_set = clusterLoader(root_dir="/home/ken/Documents/Dataset/", mode='train', dataset='imagenet', label='otsu')
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    print(len(train_set))
    img, label,_ = iter(train_loader).next()
    img = transforms.ToPILImage(mode='RGB')(img[0])
    plt.subplot(211)
    plt.imshow(img)
    plt.subplot(212)
    plt.imshow(label[0])
    plt.show()
    