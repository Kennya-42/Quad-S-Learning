import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict
from models.erfnet import ERFNet
from models.erfnet_jigsaw import ERFNet as ERFNet_jig
from models.erfnet_relpatch import ERFNet as ERFNet_relpatch
from train import Train
from test import Test
from metric.iou import IoU
import utils
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from path import get_root_path

ROOT_PATH = get_root_path()
DATASET_DIR = ROOT_PATH + "/Documents/Dataset/"
SAVE_PATH = ROOT_PATH + '/Documents/Quad-S-Learning/save/semanticSeg'
FREEZE_LR = 0.0005
FREEZE_NUM_EPOCHS = 0

def get_arguments():
    parser = ArgumentParser()
    # Execution mode
    parser.add_argument( "--mode",           choices=['train', 'test'], default='train')
    parser.add_argument( "--resume",         action='store_true')
    # Hyperparameters
    parser.add_argument( "--batch-size",     type=int,   default=6)
    parser.add_argument( "--workers",        type=int,   default=12)
    parser.add_argument( "--epochs",         type=int,   default=150)
    parser.add_argument( "--learning-rate",  type=float, default=0.0005)
    parser.add_argument( "--weight-decay",   type=float, default=1e-4)
    # Dataset
    parser.add_argument( "--dataset",        choices=['cityscapes','camvid','voc'],  default='cityscapes')
    parser.add_argument( "--height",         type=int, default=512)
    parser.add_argument( "--width",          type=int, default=1024)
    parser.add_argument( "--dataset-size",   type=int, default=-1)
    # Save
    parser.add_argument( "--name",           type=str, default='erfnet_rotcitynoaug.pth')
    parser.add_argument( "--premode",        choices=['none', 'class', 'rot', 'jig', 'kmeans', 'otsu', 'colorize','simclr','rotwdecoder','relpatch','fsjigsaw'], default='none')
    parser.add_argument( "--pretrain-name",  type=str, default='erfnet_enc_rot_city_noaug.pth')

    return parser.parse_args()

def load_dataset(dataset):
    print("Selected dataset:", args.dataset)
    print("Dataset directory:", DATASET_DIR)
    print("Save directory:", SAVE_PATH)
    train_set = dataset(root_dir=DATASET_DIR, mode='train', height=args.height, width=args.width, num_train=args.dataset_size)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_set = dataset(root_dir=DATASET_DIR, mode='val',height=args.height, width=args.width)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = val_loader
    class_encoding = train_set.color_encoding
    num_classes = len(class_encoding)
    print("Number of classes to predict:", num_classes)
    print("Number of epochs: ", args.epochs)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))
    timages, tlabels = iter(train_loader).next()
    vimages, vlabels = iter(val_loader).next()
    print("Train Image size:", timages.size())
    print("Train Label size:", tlabels.size())
    print("Val Image size:", vimages.size())
    print("Val Label size:", vlabels.size())
    class_weights = np.ones(num_classes)
    if args.dataset == 'cityscapes':
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
    elif args.dataset =='camvid':
        class_weights = utils.enet_weighing(train_loader,num_classes)
        print(class_weights)
    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float()

    return (train_loader, val_loader, test_loader), class_weights, class_encoding 
    
def train(train_loader, val_loader, class_weights, class_encoding):
    num_classes = len(class_encoding)
    ignore_index = list(class_encoding).index('unlabeled')
    if args.premode == 'none':
        print('Loading Random initialized weights')
        pretrainedEnc = None
    elif args.premode == 'class':
        print("Loading encoder pretrained on imagenet classification")
        pretrained = ERFNet(1000,classify=True)
        checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/classification/' + args.pretrain_name)
        print("Loading: ",args.pretrain_name)
        pretrained.load_state_dict(checkpnt['state_dict'])
        pretrainedEnc = pretrained.encoder
    elif args.premode == 'rot':
        print("Loading encoder pretrained on rotations")
        pretrained = ERFNet(4,classify=True)
        checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/rotation/' + args.pretrain_name)
        print("Loading: ",args.pretrain_name)
        pretrained.load_state_dict(checkpnt['state_dict'])
        pretrainedEnc = pretrained.encoder
    elif args.premode == 'jig':
        print("Loading encoder pretrained on jigsaw")
        pretrained = ERFNet_jig(num_classes=1000)
        checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/jigsaw/' + args.pretrain_name)
        print("Loading: ",args.pretrain_name)
        pretrained.load_state_dict(checkpnt['state_dict'])
        pretrainedEnc = pretrained.encoder
    elif args.premode == 'relpatch':
        print("Loading encoder pretrained on relpatch")
        pretrained = ERFNet_relpatch(num_classes=8)
        checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/relPatch/' + args.pretrain_name)
        print("Loading: ",args.pretrain_name)
        pretrained.load_state_dict(checkpnt['state_dict'])
        pretrainedEnc = pretrained.encoder
    elif args.premode == 'simclr':
        print("Loading encoder pretrained on simclr")
        pretrained = ERFNet(128,classify=True)
        checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/simclr/' + args.pretrain_name)
        print("Loading: ",args.pretrain_name)
        pretrained.load_state_dict(checkpnt['state_dict'])
        pretrainedEnc = pretrained.encoder
    elif args.premode == 'kmeans':
        print("Loading encoder pretrained on kmeans")
        model = ERFNet(4)
        checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/cluster/kmeans/' + args.pretrain_name)
        print("Loading: ",args.pretrain_name)
        model.load_state_dict(checkpnt['state_dict'])
        model.decoder.output_conv = torch.nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        freeze_model_params = utils.freezeParams(model,57)
        model = model.cuda()
    elif args.premode == 'otsu':
        print("Loading encoder pretrained on otsu")
        model = ERFNet(4)
        checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/cluster/otsu/' + args.pretrain_name)
        print("Loading: ",args.pretrain_name)
        model.load_state_dict(checkpnt['state_dict'])
        model.decoder.output_conv = torch.nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        freeze_model_params = utils.freezeParams(model,57)
        model = model.cuda()
    elif args.premode == 'colorize':
        print("Loading encoder pretrained on colorize")
        model = ERFNet(2)
        checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/colorize/' + args.pretrain_name)
        print("Loading: ",args.pretrain_name)
        model.load_state_dict(checkpnt['state_dict'])
        model.decoder.output_conv = torch.nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        freeze_model_params = utils.freezeParams(model,57)
        model = model.cuda()
    elif args.premode == 'fsjigsaw':
        print("Loading encoder pretrained on fsjigsaw")
        model = ERFNet(3)
        checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/fulljigsaw/' + args.pretrain_name)
        print("Loading: ",args.pretrain_name)
        model.load_state_dict(checkpnt['state_dict'])
        model.decoder.output_conv = torch.nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        freeze_model_params = utils.freezeParams(model,57)
        model = model.cuda()
    elif args.premode == 'rotwdecoder':
        from models.erfnet_multitask import ERFNet_multitask
        print("Loading encoder and decoder pretrained on multitask")
        pretrained = ERFNet_multitask(3)
        checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/multihead/' + args.pretrain_name)
        print("Loading: ",args.pretrain_name)
        pretrained.load_state_dict(checkpnt['state_dict'])
        model = ERFNet(num_classes,encoder=pretrained.encoder,decoder=pretrained.decoder).cuda()
        model.decoder.output_conv = torch.nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        freeze_model_params = utils.freezeParams(model,57)
        model = model.cuda()

    if args.premode != 'cluster' and args.premode != 'otsu' and args.premode != 'colorize' and args.premode != 'rotwdecoder' and args.premode != 'fsjigsaw':
        model = ERFNet(num_classes,encoder=pretrainedEnc).cuda()
        freeze_model_params = utils.freezeParams(model, 57)

    
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    metric = IoU(num_classes,ignore_index=ignore_index)
    print('Criterion: CrossEntropy')
    if FREEZE_NUM_EPOCHS:
        #preform freeze train
        print('Freeze Optimizer: Adam')
        optimizer = optim.Adam(freeze_model_params, lr=FREEZE_LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        lambda1 = lambda epoch: pow((1-((epoch-1)/FREEZE_NUM_EPOCHS)),0.9)
        lr_updater = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        freeze_train = Train(model, train_loader, optimizer, criterion, metric)
        freeze_val = Test(model, val_loader, criterion, metric)
        print('Training in Freeze mode')
        for epoch in range(FREEZE_NUM_EPOCHS):
            epoch_loss, (iou, tmiou) = freeze_train.run_epoch(epoch)
            print(">> Freeze Epoch: %d | Loss: %2.4f | Train miou: %2.4f " %(epoch, epoch_loss, tmiou))
            vepoch_loss, (iou, vmiou) = freeze_val.run_epoch()
            print(">>>> Freeze Val Epoch: %d | Loss: %2.4f | Val miou: %2.4f " %(epoch, vepoch_loss, vmiou))
            lr_updater.step(epoch)

    #train full model
    for params in model.parameters():
        params.requires_grad = True
    #try or not decoder by 10x LR
    train_params = [{'params': model.encoder.parameters(), 'lr': args.learning_rate},
                    {'params': model.decoder.parameters(), 'lr': args.learning_rate * 1}]

    print('Optimizer: Adam')
    optimizer = optim.Adam(train_params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(train_params, lr=args.learning_rate, momentum=0.9, eps=1e-08, weight_decay=args.weight_decay)
    print('Base Learning Rate: ',args.learning_rate)
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.epochs)),0.9)
    lr_updater = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    print('LR-Scheduler: poly')
    print('save name:',args.name)
    best_miou = 0
    val_miou = []
    train_miou = []
    val_loss = []
    train_loss = []
    lr_list = []

    # Start Training
    train = Train(model, train_loader, optimizer, criterion, metric)
    val = Test(model, val_loader, criterion, metric)
    vloss = 0.0
    miou = 0.0
    print('training...')
    for epoch in range(args.epochs):
        #train
        print(">> [Epoch: {0:d}] Training LR: {1:.8f}".format(epoch, lr_updater.get_lr()[0] ))
        epoch_loss, (iou, tmiou) = train.run_epoch(epoch)
        print(">> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, epoch_loss, tmiou))
        #val
        vloss, (iou, miou) = val.run_epoch()
        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, vloss, miou))
        lr_updater.step(epoch)
        if miou > best_miou:
            print("Best model thus far. Saving...")
            best_miou = miou
            utils.save_checkpoint(model, optimizer, epoch + 1, best_miou, val_miou, train_miou, val_loss, train_loss, lr_list, args, SAVE_PATH)

        train_loss.append(epoch_loss)
        train_miou.append(tmiou)
        val_loss.append(vloss)
        val_miou.append(miou)
        lr_list.append(lr_updater.get_lr()[0])
    print('Best val miou: ', best_miou)
    return model, train_loss, train_miou, val_loss, val_miou, lr_list

if __name__ == '__main__':
    assert torch.cuda.is_available(), "no GPU connected"
    args = get_arguments()
    if args.dataset == 'cityscapes':
        from data import Cityscapes as dataset
    elif args.dataset == 'camvid':
        from data import Camvid as dataset
    loaders, w_class, class_encoding = load_dataset(dataset)
    train_loader, val_loader, test_loader = loaders
    
    if args.mode.lower() == 'train':
        print("Training...")
        model, tl, tmiou, vl, vmiou, lrlist = train(train_loader, val_loader, w_class, class_encoding)
    else:
        pass
