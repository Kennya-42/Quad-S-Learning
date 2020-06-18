import torch
import argparse
import matplotlib.pyplot as plt
from PIL import Image  
from path import get_root_path
import numpy as np

ROOT_PATH = get_root_path()
DATASET_DIR = ROOT_PATH + "/Documents/Dataset/"
SAVE_PATH = ROOT_PATH + '/Documents/Quad-S-Learning/save/'

parser = argparse.ArgumentParser()
parser.add_argument( "--premode",           choices=['none', 'imagenet','rot','jig','color','cluster','rel','otsu',], default='imagenet')
parser.add_argument( "--pretrain-name",  type=str, default='erfnet_encoder.pth')
args = parser.parse_args()

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def color():
    from data import colorizeLoader as dataset
    from models.erfnet import ERFNet
    from skimage.color import lab2rgb
    train_set = dataset(root_dir=DATASET_DIR, mode='train')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    Lstack, a, b = iter(train_loader).next()
    l = Lstack[0,0]
    lab = np.zeros((224,224,3))
    lab[...,0] = l
    lab[...,1] = a
    lab[...,2] = b
    
    rgb = lab2rgb(lab)
    
    rgb = (rgb*255).astype(np.uint8)
    print(rgb)

    plt.imshow(rgb)
    plt.show()
    
def cluster():
    from data import clusterLoader as dataset
    from models.erfnet import ERFNet
    train_set = dataset(root_dir=DATASET_DIR, mode='train')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    model = ERFNet(num_classes=4).cuda()
    checkpoint = torch.load('/home/ken/Documents/Quad-S-Learning/save/cluster/erfnet_encoder.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    img,label = iter(train_loader).next()
    img = img.cuda()
    output = model(img)
    img = img.cpu()
    img,output,label = img.squeeze(),output.squeeze(),label.squeeze()
    img = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(img)
    img = img.detach().numpy()
    img = np.moveaxis(img,0,-1)
    img = (img*255).astype(np.uint8)
    output = torch.nn.Softmax(dim=0)(output)
    output = output.detach().cpu()
    _,pred = output.max(dim=0)
    plt.subplot(311)
    plt.imshow(img)
    plt.subplot(312)
    plt.imshow(pred)
    plt.subplot(313)
    plt.imshow(label)
    plt.show()

def otsu():
    from data import otsuLoader as dataset
    from models.erfnet import ERFNet
    train_set = dataset(root_dir=DATASET_DIR, mode='val')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    model = ERFNet(num_classes=4).cuda()
    checkpoint = torch.load('/home/ken/Documents/Quad-S-Learning/save/otsu/erfnet_encoder.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    img,label = iter(train_loader).next()
    img = img.cuda()
    output = model(img)
    img = img.cpu()
    img,output,label = img.squeeze(),output.squeeze(),label.squeeze()
    img = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(img)
    img = img.detach().numpy()
    img = np.moveaxis(img,0,-1)
    img = (img*255).astype(np.uint8)
    output = torch.nn.Softmax(dim=0)(output)
    output = output.detach().cpu()
    _,pred = output.max(dim=0)
    plt.subplot(311)
    plt.imshow(img)
    plt.subplot(312)
    plt.imshow(pred)
    plt.subplot(313)
    plt.imshow(label)
    plt.show()

def rel():
    from data import relpatchLoader as dataset
    from models.erfnet_relpatch import ERFNet as ERFNet_relpatch
    train_set = dataset(root_dir=DATASET_DIR, mode='val')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    model = ERFNet_relpatch(num_classes=8).cuda()
    checkpoint = torch.load('/home/ken/Documents/Quad-S-Learning/save/relPatch/erfnet_encoder.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    input,label = iter(train_loader).next()
    img = input[0,0]
    output = model(input.cuda())
    img = img.cpu()
    img,output,label = img.squeeze(),output.squeeze(),label.squeeze()
    # img = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(img)
    img = img.detach().numpy()
    img = np.moveaxis(img,0,-1)
    img = (img*255).astype(np.uint8)
    output = torch.nn.Softmax(dim=0)(output)
    output = output.detach().cpu()
    _,pred = output.max(dim=0)
    print(pred,label)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    if args.premode == 'color':
        print('testing colorize')
        color()
    elif args.premode == 'cluster':
        print('testing cluster')
        cluster()
    elif args.premode == 'otsu':
        print('testing otsu')
        otsu()
    elif args.premode == 'rel':
        print('testing relPatch')
        rel()
    else:
        print('not implement')