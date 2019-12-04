from data import Rotloader as dataset
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

DATASET_DIR = "/home/ken/Documents/Dataset/"
SAVE_PATH = '/home/ken/Documents/RealtimeSS/save'
LEARNING_RATE = 0.0005
NUM_EPOCHS = 200
BATCH_SIZE = 20 
NUM_WORKERS = 20

parser = argparse.ArgumentParser()
parser.add_argument( "--resume", action='store_true')
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
                                 
def run_train_epoch(epoch,model,criterion,optimizer,lr_updater,data_loader):
    epoch_loss = 0.0
    model.train()
    metric = accuracy()
    for step, batch_data in enumerate(data_loader):
        inputs, labels = batch_data     
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        lr_updater(optimizer, step, epoch)
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
        metric.update(outputs,labels)

    acc = metric.get_accuracy()
    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss , metric.get_accuracy()

def run_val_epoch(epoch,model,criterion,optimizer,lr_updater,data_loader):
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

    acc = metric.get_accuracy()
    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss , metric.get_accuracy()

def save_model(model, optimizer, model_path,epoch,val_acc):
    checkpoint = {
        'epoch':epoch,
        'val_acc':val_acc,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
        
    torch.save(checkpoint, model_path)

    summary_filename = SAVE_PATH+'/erfnet_rot_summary.txt'
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
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_set = dataset(root_dir=DATASET_DIR, mode='val')
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model = ERFNet(num_classes=4,classify=True).cuda()
    model_path = SAVE_PATH+'/erfnet_encoder_rot.pth'
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    lr_updater = Cust_LR_Scheduler(mode='poly', base_lr=LEARNING_RATE, num_epochs=NUM_EPOCHS,iters_per_epoch=len(train_loader))
    best_val_acc = 0
    for epoch in range(0, NUM_EPOCHS):
        print(">> [Epoch: {0:d}] Training LR: {1:.8f}".format(epoch,lr_updater.get_LR(epoch)))

        epoch_loss, train_acc = run_train_epoch(epoch, model, criterion, optimizer, lr_updater, train_loader)

        print(">> Epoch: %d | Loss: %2.4f | Train Acc: %2.4f" %(epoch,epoch_loss,train_acc))

        val_loss, val_acc = run_val_epoch(epoch, model, criterion, optimizer, lr_updater, val_loader)

        print(">>>> Val Epoch: %d | Loss: %2.4f | Val Acc: %2.4f" %(epoch, val_loss, val_acc))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print('saving best model')
            save_model(model, optimizer, model_path,epoch,val_acc)


if __name__ == '__main__':
    main()