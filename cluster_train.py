from data import clusterLoader as dataset
from models.erfnet import ERFNet
import torch.nn as nn
from train import Train
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch
import argparse
from path import get_root_path
from metric.iou import IoU

parser = argparse.ArgumentParser()
parser.add_argument( "--resume", action='store_true')
parser.add_argument( "--batch-size",     type=int,   default=6)
parser.add_argument( "--workers",        type=int,   default=12)
parser.add_argument( "--epochs",         type=int,   default=150)
parser.add_argument( "--learning-rate",  type=float, default=0.0005)
parser.add_argument( "--premode",           choices=['none', 'imagenet'], default='imagenet')
parser.add_argument( "--algorithm",           choices=['kmeans', 'otsu'], default='otsu')
parser.add_argument( "--dataset",           choices=['imagenet', 'city'], default='city')
parser.add_argument( "--savename",  type=str, default='erfnet_otsu_citywaug2.pth')
parser.add_argument( "--pretrain-name",  type=str, default='erfnet_class_imgnet.pth')
args = parser.parse_args()

#PATHS
ROOT_PATH = get_root_path()
DATASET_DIR = ROOT_PATH + "/Documents/Dataset/"
ENCODER_PATH = ROOT_PATH + '/Documents/Quad-S-Learning/save/classification/'
if args.algorithm == 'kmeans':
    SAVE_PATH = ROOT_PATH + '/Documents/Quad-S-Learning/save/cluster/kmeans/'
else:
    SAVE_PATH = ROOT_PATH + '/Documents/Quad-S-Learning/save/cluster/otsu/'
                           
def run_train_epoch(epoch,model,criterion,optimizer,data_loader):
    epoch_loss = 0.0
    model.train()
    metric = IoU(4)
    for step, batch_data in enumerate(data_loader):
        inputs, labels, _ = batch_data
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
        metric.add(outputs,labels)

    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss, metric.value()

def run_val_epoch(epoch,model,criterion,optimizer,data_loader):
    epoch_loss = 0.0
    model.eval()
    metric = IoU(4)
    for step, batch_data in enumerate(data_loader):
        with torch.no_grad():
            inputs, labels, _ = batch_data     
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            # forward propagation
            outputs = model(inputs)
            # Loss computation
            loss = criterion(outputs, labels)
            # Keep track of loss for current epoch
            epoch_loss += loss.item()
            metric.add(outputs.data, labels.data)

    iou,miou = metric.value()
    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss , metric.value()

def save_model(model, optimizer, model_path,epoch,val_miou):
    checkpoint = {
        'epoch':epoch,
        'val_miou':val_miou,
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
        summary_file.write("val miou: {0}\n". format(val_miou))
    
def main():
    train_set = dataset(root_dir=DATASET_DIR, mode='train', dataset=args.dataset, label=args.algorithm)
    val_set = dataset(root_dir=DATASET_DIR, mode='val', dataset=args.dataset, label=args.algorithm)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    timages, tlabels,_ = iter(train_loader).next()
    print('training on: ', args.dataset)
    print('initialized on: ',args.premode)
    print('train set size: ',len(train_set))
    print('val set size: ',len(val_set))
    print("Number of epochs: ", args.epochs)
    print("Train Image size:", timages.size())
    print("Train Label size:", tlabels.size())
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

    model = ERFNet(num_classes=4,classify=False,encoder=pretrainedEnc).cuda()
    model_path = SAVE_PATH + args.savename
    print('Savename: ',args.savename)
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
        best_val_miou = checkpoint['val_miou']
        print('start epoch: ', start_epoch, 'best val miou:', best_val_miou) 
    else:
        start_epoch = 0                                             
        best_val_miou = 0

    for epoch in range(start_epoch,args.epochs):
        print("> [Epoch: {0:d}] Training LR: {1:.8f}".format(epoch,lr_updater.get_lr()[0]))

        epoch_loss, (tiou,tmiou) = run_train_epoch(epoch, model, criterion, optimizer, train_loader)
        print(">> Epoch: %d | Loss: %2.4f | Train Miou: %2.4f" %(epoch,epoch_loss,tmiou))

        val_loss, (viou,vmiou) = run_val_epoch(epoch, model, criterion, optimizer, val_loader)
        print(">>>> Val Epoch: %d | Loss: %2.4f | Val Miou: %2.4f" %(epoch, val_loss, vmiou))
        lr_updater.step(epoch)
        if vmiou > best_val_miou:
            best_val_miou = vmiou
            print('saving best model')
            save_model(model, optimizer, model_path,epoch, best_val_miou)
    print(best_val_miou)
        


if __name__ == '__main__':
    main()