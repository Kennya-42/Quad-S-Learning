from models.erfnet import ERFNet as ERFNet
import torch.nn as nn
from train import Train
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
from metric.iou import IoU
import torch
import math
import argparse
import numpy as np
from path import get_root_path
from torchvision import transforms
from collections import OrderedDict
import matplotlib.pyplot as plt

ROOT_PATH = get_root_path()
DATASET_DIR = ROOT_PATH + "/Documents/Dataset/"
SAVE_PATH = ROOT_PATH + '/Documents/Quad-S-Learning/save/ed/'

parser = argparse.ArgumentParser()
parser.add_argument( "--resume", action='store_true')
parser.add_argument( "--epochs",         type=int,   default=150)
parser.add_argument( "--freeze-epochs",         type=int,   default=0)
parser.add_argument( "--batch-size",     type=int,   default=1)
parser.add_argument( "--learning-rate",  type=float, default=0.0005)
parser.add_argument( "--freeze-learning-rate",  type=float, default=0.0005)
parser.add_argument( "--workers",        type=int,   default=0)
parser.add_argument( "--encoder",           choices=['none', 'imagenet','rot'], default='imagenet')
parser.add_argument( "--decoder",           choices=['none', 'otsu','colorize'], default='otsu')
parser.add_argument( "--dataset",        choices=['cityscapes','camvid','voc'],  default='cityscapes')
parser.add_argument( "--height",         type=int, default=512)
parser.add_argument( "--width",          type=int, default=1024)
parser.add_argument( "--pretrain-enc-name",  type=str, default='erfnet_class_imgnet.pth')
parser.add_argument( "--pretrain-dec-name",  type=str, default='erfnet_encoder.pth')
parser.add_argument( "--savename",           type=str, default='erfnet_ed.pth')
args = parser.parse_args()

class LongTensorToRGBPIL(object):
    """Converts a ``torch.LongTensor`` to a ``PIL image``.
    The input is a ``torch.LongTensor`` where each pixel's value identifies the class.
    Keyword arguments:
    - rgb_encoding (``OrderedDict``): An ``OrderedDict`` that relates pixel
    values, class names, and class colors.
    """
    def __init__(self, rgb_encoding):
        if rgb_encoding is not None:
            self.rgb_encoding = rgb_encoding
        else:
            self.rgb_encoding = OrderedDict([
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

    def __call__(self, tensor):
        """Performs the conversion from ``torch.LongTensor`` to a ``PIL image``
        Keyword arguments:
        - tensor (``torch.LongTensor``): the tensor to convert
        Returns:
        A ``PIL.Image``.
        """
        # Check if label_tensor is a LongTensor
        if not isinstance(tensor, torch.LongTensor):
            raise TypeError("label_tensor should be torch.LongTensor. Got {}"
                            .format(type(tensor)))
        # Check if encoding is a ordered dictionary
        if not isinstance(self.rgb_encoding, OrderedDict):
            raise TypeError("encoding should be an OrderedDict. Got {}".format(
                type(self.rgb_encoding)))
        # label_tensor might be an image without a channel dimension, in this
        # case unsqueeze it
        if len(tensor.size()) == 2:
            tensor.unsqueeze_(0)
        color_tensor = torch.ByteTensor(3, tensor.size(1), tensor.size(2))
        for index, (class_name, color) in enumerate(self.rgb_encoding.items()):
            # if index==19:
            #     index = 255
            # Get a mask of elements equal to index
            mask = torch.eq(tensor, index).squeeze_()
            # Fill color_tensor with corresponding colors
            for channel, color_value in enumerate(color):
                color_tensor[channel].masked_fill_(mask, color_value)

        return transforms.ToPILImage()(color_tensor)


def main():
    #Prepare Data
    train_set = dataset(root_dir=DATASET_DIR, mode='train', height=args.height, width=args.width)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_set = dataset(root_dir=DATASET_DIR, mode='val', height=args.height, width=args.width)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    class_encoding = train_set.color_encoding
    num_classes = len(class_encoding)
    ignore_index = list(class_encoding).index('unlabeled')
    timages, tlabels = iter(train_loader).next()
    print('num classes: ',num_classes)
    print("Train dataset size:", len(train_set))
    print("val dataset size:", len(val_set))
    print('Batch Size: ', args.batch_size)
    print('training on: ', args.dataset)
    print("Number of epochs: ", args.epochs)
    print("Train Image size:", timages.size())
    print("Train Label size:", tlabels.size())

    #Setup Model
    model = ERFNet(num_classes=num_classes).cuda()
    checkpoint = torch.load(SAVE_PATH + args.savename)
    model.load_state_dict(checkpoint['state_dict'])
    metric = IoU(num_classes, ignore_index=ignore_index)
    model.eval()
    iou_list = []
    for vimages, vlabels in val_loader:
        with torch.no_grad():
            metric.reset()
            inputs, labels = vimages.cuda(), vlabels.cuda()
            outputs = model(inputs)
            metric.add(outputs, labels)
            iou, miou = metric.value()
            for temp in iou:
                print(temp)
            print('mIoU: ',miou)

            exit()
            iou_list.append(iou)
            if iou <0.2:
                print(iou)
                inputs,labels,outputs = inputs.cpu(),labels.cpu(),outputs.cpu()
                output = torch.nn.Softmax(dim=0)(outputs[0])
                output = output.detach()
                _,pred = output.max(dim=0)
                img = transforms.ToPILImage(mode='RGB')(inputs[0])
                label = LongTensorToRGBPIL(None)(labels[0])
                pred = LongTensorToRGBPIL(None)(pred)
                img.save('viz_res_raw_bad.png')
                label.save('viz_res_gt_bad.png')
                pred.save('viz_res_pred_bad.png')
            elif iou > 0.89:
                print(iou)
                inputs,labels,outputs = inputs.cpu(),labels.cpu(),outputs.cpu()
                output = torch.nn.Softmax(dim=0)(outputs[0])
                output = output.detach()
                _,pred = output.max(dim=0)
                img = transforms.ToPILImage(mode='RGB')(inputs[0])
                label = LongTensorToRGBPIL(None)(labels[0])
                pred = LongTensorToRGBPIL(None)(pred)
                img.save('viz_res_raw_good.png')
                label.save('viz_res_gt_good.png')
                pred.save('viz_res_pred_good.png')
    
    exit()
    inputs,labels,outputs = inputs.cpu(),labels.cpu(),outputs.cpu()
    output = torch.nn.Softmax(dim=0)(outputs[0])
    output = output.detach()
    _,pred = output.max(dim=0)
    img = transforms.ToPILImage(mode='RGB')(inputs[0])
    label = LongTensorToRGBPIL(None)(labels[0])
    pred = LongTensorToRGBPIL(None)(pred)
    # img.save('viz_res_raw_1.png')
    # label.save('viz_res_gt_1.png')
    # pred.save('viz_res_pred_1.png')
    # f, axarr = plt.subplots(1,3)
    # axarr[0].imshow(img)
    # axarr[1].imshow(label)
    # axarr[2].imshow(pred)
    # plt.show()
    exit()
   


if __name__ == '__main__':
    if args.dataset == 'cityscapes':
        from data import Cityscapes as dataset
    elif args.dataset == 'camvid':
        from data import Camvid as dataset
    main()