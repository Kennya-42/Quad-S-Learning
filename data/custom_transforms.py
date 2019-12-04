import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import math

class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,'label': mask}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        pass

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # f, axarr = plt.subplots(2,2)
        # axarr[0,0].imshow(img)
        # axarr[0,1].imshow(mask)        
        
        #TO Tensor
        img = transforms.ToTensor()(img)
        mask = np.array(mask).astype(np.int64)
        mask[mask==255] = 19
        mask = torch.from_numpy(mask)
        #back to numpy
        # img = transforms.ToPILImage(mode='RGB')(img)
        # label = mask.numpy()

        # img_tmp = np.transpose(img_tmp, axes=[1, 2, 0])
        # img_tmp *= 255
        # img_tmp = img_tmp.astype(np.uint8)

        # axarr[1,0].imshow(img)
        # axarr[1,1].imshow(mask)
        # plt.show()

        return {'image': img,'label': mask}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,'label': mask}
                
class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,'label': mask}

class RandomScaleCrop(object):
    def __init__(self, base_size=1024, crop_size=512, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:#city uses this 
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,'label': mask}

class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BICUBIC)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,'label': mask}

class RandomCrop(object):
    def __init__(self, width, height):
        self.crop_width = width
        self.crop_height = height

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        x1 = random.randint(0, w - self.crop_width)
        y1 = random.randint(0, h - self.crop_height)
        img = img.crop((x1, y1, x1 + self.crop_width, y1 + self.crop_height))
        mask = mask.crop((x1, y1, x1 + self.crop_width, y1 + self.crop_height))
        return {'image': img,'label': mask}
                
class FixedResize(object):
    def __init__(self, h,w):
        self.size = (h, w)  
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return {'image': img,'label': mask}

class FixedAspectRatioCrop(object):
    def __init__(self, size):
        self.out_size = (size,size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w,h = img.size

        if random.random() > 5.5:
            #wide
            factor = round(random.uniform(0.4, 1.0),2)
            maxh = round(min(w * factor * (3/4),h))
            maxw = round(maxh * (4/3))
            print(factor,maxw,maxh)

        else:
            #long
            factor = random.uniform(0.4, 1.0)
            factor = round(random.uniform(0.4, 1.0),2)
            maxw = round(min(h * factor * (3/4),w))
            maxh = round(maxw * (4/3))
            print(factor,maxw,maxh)
            

        x = random.randint(0, w - maxw)
        print(w-maxw,x)
        y = random.randint(0, h - maxh)
                
        img = img.crop((x, y, x+maxw, y+maxh))
        mask = mask.crop((x, y, x+maxw, y+maxh))
        img = img.resize(self.out_size, Image.BILINEAR)
        mask = mask.resize(self.out_size, Image.NEAREST)
        return {'image': img,'label': mask}

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return {'image': img,'label': mask}

class RandomTranslation(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        transX = random.randint(-4, 4) 
        transY = random.randint(-4, 4)
        img = ImageOps.expand(img, border=(transX,transY,0,0), fill=0)
        mask = ImageOps.expand(mask, border=(transX,transY,0,0), fill=255) #pad label filling with 255
        img = img.crop((0, 0, img.size[0]-transX, img.size[1]-transY))
        mask = mask.crop((0, 0, mask.size[0]-transX, mask.size[1]-transY))
        return {'image': img,'label': mask}

if __name__ == "__main__":
    img = Image.open("/home/ken/Documents/RealtimeSS/test_content/aachen_000000_000019_leftImg8bit.png")
    mask = Image.open("/home/ken/Documents/RealtimeSS/test_content/aachen_000000_000019_gtFine_labelIds.png")
    sample = {'image': img, 'label': mask}
    sample = FixedAspectRatioCrop(size=512)(sample)
    img , mask = sample['image'],sample['label']
    f, axarr = plt.subplots(2,1)
    axarr[0].imshow(img)
    axarr[1].imshow(mask)
    plt.show()
