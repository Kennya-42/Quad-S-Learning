from data import simclrloader as dataset
from models.erfnet import ERFNet
# from models.erfnet_simclr import ERFNet as ERFnet_simclr
import torch.nn as nn
from train import Train
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch
import math
import argparse
from path import get_root_path
import os
from ntxent_loss import NTXentLoss
# os.environ['cuda_launch_blocking'] = '1'

ROOT_PATH = get_root_path()
DATASET_DIR = ROOT_PATH + "/Documents/Dataset/"
ENCODER_PATH = ROOT_PATH + '/Documents/Quad-S-Learning/save/classification/'
SAVE_PATH = ROOT_PATH + '/Documents/Quad-S-Learning/save/simclr/'

parser = argparse.ArgumentParser()
parser.add_argument( "--resume", action='store_true')
parser.add_argument( "--premode",           choices=['none', 'imagenet'], default='imagenet')
parser.add_argument( "--dataset",           choices=['imagenet', 'city'], default='imagenet')
parser.add_argument( "--batch-size",     type=int,   default=128)
parser.add_argument( "--workers",        type=int,   default=32)
parser.add_argument( "--epochs",         type=int,   default=150)
parser.add_argument( "--learning-rate",  type=float, default=0.0005)
parser.add_argument( "--weight-decay",   type=float, default=1e-4)
parser.add_argument( "--pretrain-name",  type=str, default='erfnet_class_imgnet.pth')
parser.add_argument( "--savename",           type=str, default='erfnet_simclr_imagenet.pth')
args = parser.parse_args()
            
def run_train_epoch(epoch,model,criterion,optimizer,data_loader):
    epoch_loss = 0.0
    model.train()
    for step, batch_data in enumerate(data_loader):
        xis, xjs = batch_data
        xis, xjs = Variable(xis).cuda(), Variable(xjs).cuda()
        #Forward Propagation
        zis = model(xis)
        zjs = model(xjs)
        #normalize
        zis = torch.nn.functional.normalize(zis, dim=1)
        zjs = torch.nn.functional.normalize(zjs, dim=1)
        # Loss computation
        loss = criterion(zis, zjs)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Keep track of loss for current epoch
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss

def run_val_epoch(epoch,model,criterion,data_loader):
    epoch_loss = 0.0
    model.eval()
    for step, batch_data in enumerate(data_loader):
        with torch.no_grad():
            xis, xjs = batch_data
            xis, xjs = Variable(xis).cuda(), Variable(xjs).cuda()
            #Forward Propagation
            zis = model(xis)
            zjs = model(xjs)
            #normalize
            zis = torch.nn.functional.normalize(zis, dim=1)
            zjs = torch.nn.functional.normalize(zjs, dim=1)
            # Loss computation
            loss = criterion(zis, zjs)
            # Keep track of loss for current epoch
            epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss 

def save_model(model, optimizer, model_path,epoch, val_loss):
    checkpoint = {
        'epoch': epoch,
        'val_loss': val_loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
        
    torch.save(checkpoint, model_path)

    summary_filename = SAVE_PATH + args.savename + '_summary.txt'
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)
        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("val_loss: {0}\n". format(val_loss))
    
def main():
    #Prepare Data
    train_set = dataset(root_dir=DATASET_DIR, mode='train',dataset=args.dataset)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
    val_set = dataset(root_dir=DATASET_DIR, mode='val',dataset=args.dataset)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
    timages, tlabels = iter(train_loader).next()
    print('training on: ', args.dataset)
    print('initialized on: ',args.premode)
    print('train set size: ',len(train_set))
    print('val set size: ',len(val_set))
    print("Number of epochs: ", args.epochs)
    print("Train Image size:", timages.size())
    print("Train Label size:", tlabels.size())
    print('save name: ',args.savename)
    #Load Model
    if args.premode == 'none':
        pretrainedEnc = None
    elif args.premode == 'imagenet':
        print("Loading encoder pretrained on imagenet classification")
        pretrained = ERFNet(1000,classify=True)
        checkpnt = torch.load(ENCODER_PATH + args.pretrain_name)
        print("Loading: ",args.pretrain_name)
        pretrained.load_state_dict(checkpnt['state_dict'])
        pretrainedEnc = pretrained.encoder

    model = ERFNet(num_classes=128, classify=True, encoder=pretrainedEnc).cuda()
    model_path = SAVE_PATH + args.savename
    criterion = NTXentLoss("cuda", args.batch_size, 0.5, True)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.epochs)),0.9)
    lr_updater = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    if args.resume:
        print('resuming from checkpoint')
        checkpoint = torch.load(SAVE_PATH + args.savename)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        best_val_loss = checkpoint['val_loss']
        print('start epoch: ', start_epoch, 'best val loss:', best_val_loss) 
    else:
        start_epoch = 0                                             
        best_val_loss = 0
    print("Training!")  
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = run_train_epoch(epoch, model, criterion, optimizer, train_loader)
        print(">> Epoch: %d | Loss: %2.4f" %(epoch, epoch_loss))
        val_loss = run_val_epoch(epoch, model, criterion, val_loader)
        print(">>>> Val Epoch: %d | Loss: %2.4f " %(epoch, val_loss))
        lr_updater.step(epoch)
        if val_loss < best_val_loss or epoch == 0:
            best_val_loss = val_loss
            print('saving best model')
            save_model(model, optimizer, model_path, epoch, best_val_loss)
    print(best_val_loss)

if __name__ == '__main__':
    main()