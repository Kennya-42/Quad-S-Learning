import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from lr_scheduler import Cust_LR_Scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict
from models.enet import ENet
from models.erfnet import ERFNet
from train import Train
from test import Test
from metric.iou import iou
import utils
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

args = get_arguments()

def get_arguments():
    parser = ArgumentParser()
    # Execution mode
    parser.add_argument( "--mode",           choices=['train', 'test'], default='train')
    parser.add_argument( "--model",          choices=['ERFnet', 'Deeplab'], default='ERFnet')
    parser.add_argument( "--resume",         action='store_true')
    # Hyperparameters
    parser.add_argument( "--batch-size",     type=int,   default=5)
    parser.add_argument( "--val-batch-size", type=int,   default=5)
    parser.add_argument( "--workers",        type=int,   default=0)
    parser.add_argument( "--epochs",         type=int,   default=500)
    parser.add_argument( "--learning-rate",  type=float, default=0.0005)
    parser.add_argument( "--weight-decay",   type=float, default=1e-4)
    # Dataset
    parser.add_argument( "--dataset",        choices=['cityscapes','kitti'],  default='cityscapes')
    parser.add_argument( "--dataset-dir",    type=str, default="/home/ken/Documents/Dataset/")
    parser.add_argument( "--height",         type=int, default=512)
    parser.add_argument( "--width",          type=int, default=1024)
    
    parser.add_argument( "--name",           type=str, default='ERFnet.pth')
    parser.add_argument( "--pretrain_name",  type=str, default='erfnet_encoder_pretrained.pth')
    parser.add_argument( "--save-dir",       type=str, default='save/ERFnet')

    return parser.parse_args()

def load_dataset(dataset):
    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    train_set = dataset(root_dir=args.dataset_dir, mode='train', height=args.height, width=args.width)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_set = dataset(root_dir=args.dataset_dir, mode='val',height=args.height, width=args.width)
    val_loader = data.DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers)
    test_loader = val_loader

    class_encoding = train_set.color_encoding
    num_classes = len(class_encoding)
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    timages, tlabels = iter(train_loader).next()
    vimages, vlabels = iter(val_loader).next()

    print("Train Image size:", timages.size())
    print("Train Label size:", tlabels.size())
    print("Val Image size:", vimages.size())
    print("Val Label size:", vlabels.size())
    
    class_weights = np.ones(num_classes)
    class_weights[0] = 2.8149201869965	
    class_weights[1] = 6.9850029945374	
    class_weights[2] = 3.7890393733978	
    class_weights[3] = 9.9428062438965	
    class_weights[4] = 9.7702074050903	
    class_weights[5] = 9.5110931396484	
    class_weights[6] = 10.311357498169	
    class_weights[7] = 10.026463508606	
    class_weights[8] = 4.6323022842407	
    class_weights[9] = 9.5608062744141	
    class_weights[10] = 7.8698215484619	
    class_weights[11] = 9.5168733596802	
    class_weights[12] = 10.373730659485	
    class_weights[13] = 6.6616044044495	
    class_weights[14] = 10.260489463806	
    class_weights[15] = 10.287888526917	
    class_weights[16] = 10.289801597595	
    class_weights[17] = 10.405355453491	
    class_weights[18] = 10.138095855713	
    class_weights[19] = 0
    print(class_weights)
    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float()

    return (train_loader, val_loader, test_loader), class_weights, class_encoding 
    
def train(train_loader, val_loader, class_weights, class_encoding):
    print("Training...")
    num_classes = len(class_encoding)
    # pick model
    print("Loading encoder pretrained in imagenet")

    pretrainedEnc = ERFNet(1000,classify=True)
    checkpnt = torch.load('save/erfnet_encoder.pth')
    pretrainedEnc.load_state_dict(checkpnt['state_dict'])

    if args.model.lower() == 'erfnet':
        print("Model Name: ERFnet")
        model = ERFNet(num_classes,encoder=pretrainedEnc.encoder)
        train_params = model.parameters()
    
    # Define Optimizer
    if args.optimizer == 'SGD':
        print('Optimizer: SGD')
        optimizer = torch.optim.SGD(train_params, momentum=0.9, weight_decay=args.weight_decay)
    else:
        print('Optimizer: Adam')
        optimizer = optim.Adam(train_params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    print('Base Learning Rate: ',args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None

    metric = IoU(num_classes, ignore_index=ignore_index)

    model = model.cuda()
    criterion = criterion.cuda()
    
    # lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay)
    lr_updater = Cust_LR_Scheduler(mode='poly', base_lr=args.learning_rate, num_epochs=args.epochs,iters_per_epoch=len(train_loader), lr_step=0,warmup_epochs=1)
    #resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_miou, val_miou, train_miou, val_loss, train_loss, lr_list = utils.load_checkpoint( model,optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} | Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0
        val_miou = []
        train_miou = []
        val_loss = []
        train_loss = []
        lr_list = []

    # Start Training
    train = Train(model, train_loader, optimizer, criterion, metric,lr_updater)
    val = Test(model, val_loader, criterion, metric)
    vloss = 0.0
    miou = 0.0   
    for epoch in range(start_epoch, args.epochs):
        print(">> [Epoch: {0:d}] Training LR: {1:.8f}".format(epoch,lr_updater.get_LR(epoch)))
        # lr_updater.step()
        epoch_loss, (iou, tmiou) = train.run_epoch(epoch)
        if vloss == 0.0:
            vloss = epoch_loss
        print(">> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, epoch_loss, tmiou))
        #preform a validation test
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))
            vloss, (iou, miou) = val.run_epoch()
            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, vloss, miou))
            # Save the model if it's the best thus far
            if miou > best_miou:
                print("Best model thus far. Saving...")
                best_miou = miou
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou, val_miou, train_miou, val_loss, train_loss, lr_list, args)

        train_loss.append(epoch_loss)
        train_miou.append(tmiou)
        val_loss.append(vloss)
        val_miou.append(miou)
        lr_list.append(lr_updater.get_LR(epoch))

    return model, train_loss, train_miou, val_loss, val_miou,lr_list

def test(model, test_loader, class_weights, class_encoding):
    print("Testing...")
    num_classes = len(class_encoding)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if use_cuda:
        criterion = criterion.cuda()

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, metric, use_cuda)

    print(">>>> Running test dataset")
    loss, (iou, miou) = test.run_epoch(args.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))
    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))


if __name__ == '__main__':
    
    assert torch.cuda.is_available(), "no GPU connected"

    if args.dataset.lower() == 'cityscapes':
        from data import Cityscapes as dataset
    elif args.dataset.lower() == 'kitti':
        from data import Kitti as dataset
    else:
        raise RuntimeError("\"{0}\" is not a supported dataset.".format(args.dataset))
    
    loaders, w_class, class_encoding = load_dataset(dataset)
    train_loader, val_loader, test_loader = loaders
    
    if args.mode.lower() == 'train':
        from matplotlib.pyplot import figure
        figure(num=None, figsize=(16, 12), dpi=250, facecolor='w', edgecolor='k')
        model,tl,tmiou,vl,vmiou,lrlist = train(train_loader, val_loader, w_class, class_encoding)
    elif args.mode.lower() == 'test':
        num_classes = len(class_encoding)
        if args.model.lower() == 'erfnet':
            model = ERFNet(num_classes)
        else:
            model = DeepLab(num_classes)
        model = model.cuda()
        optimizer = optim.Adam(model.parameters())
        model = utils.load_checkpoint(model, optimizer, args.save_dir, args.name)[0]
        test(model, test_loader, w_class, class_encoding)
    else:
        raise RuntimeError("\"{0}\" is not a valid choice for execution mode.".format(args.mode))
