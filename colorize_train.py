from data import colorizeLoader as dataset
from models.erfnet import ERFNet
import torch.nn as nn
from train import Train
from lr_scheduler import Cust_LR_Scheduler
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch
import math
import argparse
from path import get_root_path

ROOT_PATH = get_root_path()
DATASET_DIR = ROOT_PATH + "/Documents/Dataset/"
SAVE_PATH = ROOT_PATH + '/Documents/Quad-S-Learning/save/'
LEARNING_RATE = 0.00005
NUM_EPOCHS = 200
BATCH_SIZE = 2 
NUM_WORKERS = 0

parser = argparse.ArgumentParser()
parser.add_argument( "--resume", action='store_true')
parser.add_argument( "--premode",           choices=['none', 'imagenet'], default='imagenet')
parser.add_argument( "--pretrain-name",  type=str, default='erfnet_encoder.pth')
args = parser.parse_args()

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
               
def run_train_epoch(epoch,model,criterion,optimizer,lr_updater,data_loader):
    epoch_loss = 0.0
    model.train()
    metric = AverageMeter()
    for step, batch_data in enumerate(data_loader):
        input, A, B = batch_data    
        input, A, B = Variable(input).cuda(), Variable(A).cuda(), Variable(B).cuda()
        lr_updater(optimizer, step, epoch)
        groundTruth = torch.cat((A,B),1)
        #Forward Propagation
        outputs = model(input)
        # Loss computation
        loss = criterion(outputs, groundTruth)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Keep track of loss for current epoch
        epoch_loss += loss.item()
        metric.update(loss.item(),BATCH_SIZE)
    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss, metric.avg 

def run_val_epoch(epoch, model, criterion, optimizer, data_loader):
    epoch_loss = 0.0
    model.eval()
    metric = AverageMeter()
    for step, batch_data in enumerate(data_loader):
        with torch.no_grad():
            input, A, B = batch_data    
            input, A, B = Variable(input).cuda(), Variable(A).cuda(), Variable(B).cuda()
            groundTruth = torch.cat((A,B),1)
            #Forward Propagation
            outputs = model(input)
            # Loss computation
            loss = criterion(outputs, groundTruth)
            # Keep track of loss for current epoch
            epoch_loss += loss.item()
            metric.update(loss.item(),BATCH_SIZE)
    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss , metric.avg

def save_model(model, optimizer, model_path,epoch,val_avg):
    checkpoint = {
        'epoch':epoch,
        'val_avg_loss':val_avg,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
        
    torch.save(checkpoint, model_path)

    summary_filename = SAVE_PATH+'colorize/erfnet_colorize_summary.txt'
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)
        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("Val Avg Loss: {0}\n". format(val_avg))
        summary_file.write("Batch_size: {0}\n". format(BATCH_SIZE))
        summary_file.write("Learning Rate: {0}\n". format(LEARNING_RATE))
    
def main():
    train_set = dataset(root_dir=DATASET_DIR, mode='train')
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_set = dataset(root_dir=DATASET_DIR, mode='val')
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    #Load Model
    if args.premode == 'none':
        pretrainedEnc = None
    elif args.premode == 'imagenet':
        print("Loading encoder pretrained on imagenet classification")
        pretrained = ERFNet(1000,classify=True)
        checkpnt = torch.load(SAVE_PATH+ 'Classification/' + args.pretrain_name)
        print("Loading: ",args.pretrain_name)
        pretrained.load_state_dict(checkpnt['state_dict'])
        pretrainedEnc = pretrained.encoder

    model = ERFNet(num_classes=2,encoder=pretrainedEnc).cuda()
    model_path = SAVE_PATH+'colorize/erfnet_encoder_colorize.pth'
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    lr_updater = Cust_LR_Scheduler(mode='poly', base_lr=LEARNING_RATE, num_epochs=NUM_EPOCHS,iters_per_epoch=len(train_loader))
    best_val_avg_loss = 0
    for epoch in range(0, NUM_EPOCHS):
        print(">> [Epoch: {0:d}] Training LR: {1:.8f}".format(epoch,lr_updater.get_LR(epoch)))

        epoch_loss, avg_loss = run_train_epoch(epoch, model, criterion, optimizer, lr_updater, train_loader)

        print(">> Epoch: %d | Loss: %2.4f | Avg Loss: %2.4f" %(epoch,epoch_loss,avg_loss))
        
        val_loss, val_avg_loss = run_val_epoch(epoch, model, criterion, optimizer, val_loader)

        print(">>>> Val Epoch: %d | Loss: %2.4f | Val Avg Loss: %2.4f" %(epoch, val_loss, val_avg_loss))
        
        if val_avg_loss > best_val_avg_loss:
            best_val_avg_loss = val_avg_loss
            print('saving best model')
            save_model(model, optimizer, model_path,epoch,val_avg_loss)


if __name__ == '__main__':
    main()