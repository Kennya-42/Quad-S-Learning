from data import Imagenet as dataset
from models.erfnet import ERFNet
import torch.nn as nn
from train import Train
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch
import argparse
from path import get_root_path

ROOT_PATH = get_root_path()
DATASET_DIR = ROOT_PATH + "/Documents/Dataset/"
SAVE_PATH = ROOT_PATH + '/Documents/Quad-S-Learning/save/classification/'

parser = argparse.ArgumentParser()
parser.add_argument( "--resume", action='store_true')
parser.add_argument( "--batch-size",     type=int,   default=40)
parser.add_argument( "--workers",        type=int,   default=0)
parser.add_argument( "--epochs",         type=int,   default=150)
parser.add_argument( "--learning-rate",  type=float, default=0.0005)
parser.add_argument( "--savename",           type=str, default='erfnet_simclr_imagenet.pth')
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
                                 
def run_train_epoch(epoch, model, criterion, optimizer, data_loader):
    epoch_loss = 0.0
    model.train()
    metric = accuracy()
    for step, batch_data in enumerate(data_loader):
        inputs, labels = batch_data     
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        #Forward Propagation
        outputs = model(inputs)
        # Loss computation
        loss = criterion(outputs, labels)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Keep track of loss for current epoch
        epoch_loss += loss.item()
        metric.update(outputs, labels)

    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss , metric.get_accuracy()

def run_val_epoch(epoch, model, criterion, optimizer, data_loader):
    epoch_loss = 0.0
    model.eval()
    metric = accuracy()
    for step, batch_data in enumerate(data_loader):
        with torch.no_grad():
            inputs, labels = batch_data     
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            # forward propagation
            outputs = model(inputs)
            # Loss computation
            loss = criterion(outputs, labels)
            # Keep track of loss for current epoch
            epoch_loss += loss.item()
            metric.update( outputs, labels)

    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss , metric.get_accuracy()

def save_model(model, optimizer, model_path, epoch, val_acc):
    checkpoint = {
        'epoch':epoch,
        'val_acc':val_acc,
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
        summary_file.write("Val Acc: {0}\n". format(val_acc))
    
def main():
    train_set = dataset(root_dir=DATASET_DIR, mode='train')
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_set = dataset(root_dir=DATASET_DIR, mode='val')
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print('train set size: ',len(train_set))
    print("Number of epochs: ", args.epochs)
    print('batch size: ',args.batch_size)
    print('learning Rate:', args.learning_rate)
    model = ERFNet(num_classes=1000, classify=True).cuda()
    model_path = SAVE_PATH + args.savename
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.epochs)),0.9)
    lr_updater = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    if args.resume:
        print('resuming from checkpoint')
        checkpoint = torch.load(SAVE_PATH + args.savename)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        best_val_acc = checkpoint['val_acc']
        print('start epoch: ', start_epoch, 'best val acc:', best_val_acc) 
    else:
        start_epoch = 0                                             
        best_val_acc = 0

    for epoch in range(start_epoch,args.epochs):
        print(">> [Epoch: {0:d}] Training LR: {1:.8f}".format(epoch, lr_updater.get_lr()[0]))
        epoch_loss, train_acc = run_train_epoch(epoch, model, criterion, optimizer, train_loader)
        print(">> Epoch: %d | Loss: %2.4f | Train Acc: %2.4f" %(epoch, epoch_loss, train_acc))
        val_loss, val_acc = run_val_epoch(epoch, model, criterion, optimizer, val_loader)
        print(">>>> Val Epoch: %d | Loss: %2.4f | Val Acc: %2.4f" %(epoch, val_loss, val_acc))
        lr_updater.step(epoch)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print('saving best model')
            save_model(model, optimizer, model_path, epoch, val_acc)
    print(best_val_acc)


if __name__ == '__main__':
    main()