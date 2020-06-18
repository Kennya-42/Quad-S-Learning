import torch
import argparse
from path import get_root_path
from metric.iou import IoU
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument( "--mode",           choices=['cluster', 'seg','colorize','fsjigsaw'], default='cluster')
parser.add_argument( "--algorithm",           choices=['kmeans', 'otsu'], default='kmeans')
parser.add_argument( "--dataset",           choices=['imagenet', 'city'], default='imagenet')
parser.add_argument( "--pretrain-name",  type=str, default='erfnet_class_imgnet.pth')
args = parser.parse_args()

ROOT_PATH = get_root_path()
DATASET_DIR = ROOT_PATH + "/Documents/Dataset/"

def cluster():
    NUM_CLASSES = 4
    from data import clusterLoader as dataset
    from models.erfnet import ERFNet
    train_set = dataset(root_dir=DATASET_DIR, mode='train', dataset=args.dataset, label=args.algorithm)
    val_set = dataset(root_dir=DATASET_DIR, mode='val', dataset=args.dataset, label=args.algorithm)
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=0)
    image, label, image_un = iter(train_loader).next()
    print('Dataset: ', args.dataset)
    print('train set size: ',len(train_set))
    print("Train Image size:", image.size())
    print("Train Label size:", label.size())
    model = ERFNet(num_classes=4,classify=False,encoder=None).cuda()
    checkpnt = torch.load(ROOT_PATH + '/Documents/Quad-S-Learning/save/cluster/' + args.algorithm + '/' + args.pretrain_name)
    print("Loading: ",args.pretrain_name)
    model.load_state_dict(checkpnt['state_dict'])
    image = image.cuda()
    img = transforms.ToPILImage(mode='RGB')(image_un[0].cpu())
    pred = model(image).cpu()
    print(pred.size())
    pred = nn.Softmax(dim=1)(pred)
    pred = torch.argmax(pred, dim=1)
    print(pred.size())
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(pred.squeeze())
    plt.subplot(133)
    plt.imshow(label.squeeze())
    plt.show()

def colorize():
    from data import colorizeLoader as dataset
    from models.erfnet import ERFNet
    from skimage.color import lab2rgb, rgb2lab
    train_set = dataset(root_dir=DATASET_DIR, mode='train',dataset=args.dataset)
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    val_set = dataset(root_dir=DATASET_DIR, mode='val',dataset=args.dataset)
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=0)
    img_gray, img_ab, img = iter(train_loader).next()
    print('training on: ', args.dataset)
    print('train set size: ',len(train_set))
    print("Train Image size:", img_gray.size())
    print("Train Label size:", img_ab.size())
    model = ERFNet(num_classes=2).cuda()
    checkpnt = torch.load(ROOT_PATH + '/Documents/Quad-S-Learning/save/colorize/'  + args.pretrain_name)
    print("Loading: ",args.pretrain_name)
    model.load_state_dict(checkpnt['state_dict'])
    pred = model(img_gray.cuda()).cpu().squeeze()
    img_gray = img_gray.cpu().squeeze()[0]+50
    print(pred.size())
    print(img_gray.size())
    reconstructed = torch.cat([img_gray.unsqueeze(dim = 0),pred],dim = 0)
    print(reconstructed.size())
    reconstructed = reconstructed.detach().permute(1, 2, 0)
    print(reconstructed.size())
    reconstructed = lab2rgb(reconstructed)
    plt.subplot(131)
    plt.imshow(img.squeeze())
    plt.subplot(132)
    plt.imshow(img_gray,cmap='gray')
    plt.subplot(133)
    plt.imshow(reconstructed)
    plt.show()

def fsjigsaw():
    from data import fulljigsawLoader as dataset
    from models.erfnet import ERFNet
    train_set = dataset(root_dir=DATASET_DIR,perm_dir=ROOT_PATH+'/Documents/Quad-S-Learning/data', mode='train')
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    val_set = dataset(root_dir=DATASET_DIR, perm_dir=ROOT_PATH+'/Documents/Quad-S-Learning/data', mode='val')
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=0)
    img, label = iter(train_loader).next()
    print('training on: ', args.dataset)
    print('train set size: ',len(train_set))
    print('val set size: ',len(val_set))
    print("Train Image size:", img.size())
    print("Train Label size:", label.size())
    #test model
    model = ERFNet(num_classes=3).cuda()
    checkpnt = torch.load(ROOT_PATH + '/Documents/Quad-S-Learning/save/fulljigsaw/'  + args.pretrain_name)
    print("Loading: ",args.pretrain_name)
    model.load_state_dict(checkpnt['state_dict'])
    pred = model(img.cuda()).cpu()
    img = img.cpu()

    img = transforms.ToPILImage(mode='RGB')(img.squeeze())
    label = transforms.ToPILImage(mode='RGB')(label.squeeze())
    pred = transforms.ToPILImage(mode='RGB')(pred.squeeze())
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(pred)
    plt.subplot(133)
    plt.imshow(label)
    plt.show()
    

if __name__ == '__main__':
    if args.mode == 'cluster':
        cluster()
    elif args.mode == 'colorize':
        colorize()
    elif args.mode == 'fsjigsaw':
        fsjigsaw()
        