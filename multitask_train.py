from models.erfnet_multitask import ERFNet_multitask
from models.erfnet import ERFNet as ERFNet_normal
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
import os
os.environ['cuda_launch_blocking'] = '1'

ROOT_PATH = get_root_path()
DATASET_DIR = ROOT_PATH + "/Documents/Dataset/"
SAVE_PATH = ROOT_PATH + '/Documents/Quad-S-Learning/save/multihead/'

parser = argparse.ArgumentParser()
parser.add_argument( "--resume", action='store_true')
parser.add_argument( "--premode",           choices=['none', 'imagenet'], default='imagenet')
parser.add_argument( "--method",           choices=['rot', 'jig'], default='jig')
parser.add_argument( "--pretrain-name",  type=str, default='erfnet_class_imgnet.pth')
parser.add_argument( "--savename",           type=str, default='erfnet_jig_wreconstruct.pth')
parser.add_argument( "--batch-size",     type=int,   default=64)
parser.add_argument( "--workers",        type=int,   default=12)
parser.add_argument( "--epochs",         type=int,   default=150)
parser.add_argument( "--learning-rate",  type=float, default=0.0005)
args = parser.parse_args()

class accuracy():
    def __init__(self):
        self.num_correct = 0
        self.total = 0
    def update(self,output,label):
        _,pred = torch.max(output, 1)
        c = (pred == label).squeeze()
        self.num_correct += c.sum().item()
        self.total += output.size()[0]
    def get_accuracy(self):
        acc = self.num_correct / self.total
        return acc
                             
def run_train_epoch(epoch, model, criterion, criterion2, optimizer, data_loader, loss_adjust):
    epoch_loss = 0.0
    # print(loss_adjust)
    loss_adjust = [1,1]
    model.train()
    metric = accuracy()
    for step, batch_data in enumerate(data_loader):
        inputs, labels, labels2 = batch_data     
        inputs, labels, labels2 = inputs.cuda(), labels.cuda(), labels2.cuda()
        optimizer.zero_grad()
        #classify rotation
        out1, out2 = model(inputs)
        # _,pred = torch.max(out1, 1)
        # print(pred)
        # print(labels)
       
        loss1 = criterion(out1, labels) #* loss_adjust[0]
        # print(loss1)
        #reconstruct image
        loss2 = criterion2(out2, labels2) #* loss_adjust[1]
        # print(loss2)
        # import matplotlib.pyplot as plt
        # from torchvision import transforms
        # plt.subplot(311)
        # plt.imshow(transforms.ToPILImage(mode='RGB')(inputs[0].cpu()))
        # plt.subplot(312)
        # plt.imshow(transforms.ToPILImage(mode='RGB')(labels2[0].cpu()))
        # plt.subplot(313)
        # plt.imshow(transforms.ToPILImage(mode='RGB')(out2[0].cpu()))
        # plt.show()
        loss = loss1 + loss2
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
        # print(epoch_loss)
        metric.update(out1,labels)
        # exit()

    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss, metric.get_accuracy()

def run_val_epoch(epoch, model, criterion, criterion2, data_loader):
    epoch_loss = 0.0
    model.eval()
    metric = accuracy()
    for step, batch_data in enumerate(data_loader):
        with torch.no_grad():
            inputs, labels, labels2 = batch_data
            inputs, labels, labels2 = inputs.cuda(), labels.cuda(), labels2.cuda()
            #classify rotation
            out1, out2 = model(inputs)
            loss1 = criterion(out1, labels) #* loss_adjust[0]
            #reconstruct image
            loss2 = criterion2(out2, labels2) #* loss_adjust[1]
            loss = loss1 + loss2
            epoch_loss += loss.item()
            metric.update(out1,labels)
    
    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss, metric.get_accuracy()

def save_model(model, optimizer, model_path, epoch, val_best_enc):
    print('saving best model')
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_best_enc': val_best_enc}
        
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
        summary_file.write("val_best_enc: {0}\n". format(val_best_enc))
    
def main():
    if args.method == "rot":
        from data import Rotloader as dataset
        num_classes = 4
    else:
        from data import fulljigsawLoader as dataset
        num_classes = 1000
    train_set = dataset(root_dir=DATASET_DIR, mode='train', dataset='city', include_extralabel=True)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_set = dataset(root_dir=DATASET_DIR, mode='val', dataset='city', include_extralabel=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print('batchsize: ',args.batch_size)
    print('train size: ',len(train_set))
    print('val size: ',len(val_set))
    
    #Load Model
    if args.premode == 'none':
        pretrainedEnc = None
    elif args.premode == 'imagenet':
        print("Loading encoder pretrained on imagenet classification")
        pretrained = ERFNet_normal(1000,classify=True)
        checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/classification/' + args.pretrain_name)
        print("Loading: ",args.pretrain_name)
        pretrained.load_state_dict(checkpnt['state_dict'])
        pretrainedEnc = pretrained.encoder
        
    #Setup Model
    model = ERFNet_multitask(num_classes_dec=3,num_classes_enc=num_classes,encoder=pretrainedEnc).cuda()
    model_path = SAVE_PATH + args.savename
    print('Criterion: CrossEntropy')
    print('Criterion2: MSE')
    criterion = nn.CrossEntropyLoss().cuda()
    criterion2 = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.epochs)),0.9)
    lr_updater = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    best_val_loss = 0
    print('training....')
    #>>>>>>>>>>>>>>>TRAIN<<<<<<<<<<<<<<<<
    for epoch in range(args.epochs):
        loss_adjust = [(np.cos(epoch/100*np.pi)+1)/2, (np.cos((epoch+50)/100*np.pi)+1)/2]
        epoch_loss, train_acc = run_train_epoch(epoch, model, criterion, criterion2, optimizer, train_loader, loss_adjust)
        print(">> Epoch: %d | Loss: %2.4f | Train Acc: %2.4f" %(epoch, epoch_loss, train_acc))
        val_loss, val_acc = run_val_epoch(epoch, model, criterion, criterion2, val_loader)
        print(">>>> Val Epoch: %d | Loss: %2.4f | Val Acc: %2.4f" %(epoch, val_loss, val_acc ))
        lr_updater.step(epoch)

        if val_loss < best_val_loss or epoch == 0:
            best_val_loss = val_loss
            save_model(model, optimizer, model_path, epoch, best_val_loss)


if __name__ == '__main__':
    main()