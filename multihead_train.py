from data import Rotloader as dataset_rot
from data import Cityscapes as dataset_cityscapes
from models.erfnet_multihead import ERFNet
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
SAVE_PATH = ROOT_PATH + '/Documents/Quad-S-Learning/save/multihead'
LEARNING_RATE = 0.0005
NUM_EPOCHS = 150
BATCH_SIZE = 4 
NUM_WORKERS = 6
FREEZE_LR = 0.005
FREEZE_NUM_EPOCHS = 0

parser = argparse.ArgumentParser()
parser.add_argument( "--resume", action='store_true')
parser.add_argument( "--premode",           choices=['none', 'imagenet'], default='imagenet')
parser.add_argument( "--pretrain-name",  type=str, default='erfnet_class_imgnet.pth')
parser.add_argument( "--name",           type=str, default='erfnet_rot_ss.pth')
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

def get_class_weights():
    class_weights = np.ones(20)
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
    class_weights = torch.from_numpy(class_weights).float()
    return class_weights

def freezeParams(model,cnt):
    model_params = []
    model_params.extend(model.parameters())
    for i, param in enumerate(model_params):
        if i > cnt:
            param.requires_grad = False
    
    return model_params
                                 
def run_train_epoch(epoch, model, criterion, criterion2, optimizer, metric, data_loader_seg, data_loader_rot, loss_adjust):
    epoch_loss = 0.0
    # print(loss_adjust)
    model.train()
    metric.reset()
    metric2 = accuracy()
    
    for step, batch_data in enumerate(data_loader_seg):
        inputs, labels = batch_data     
        # inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        #semantic seg forward and loss
        outputs = model(inputs, mode='full')
        loss1 = criterion(outputs, labels) * loss_adjust[0]
        # loss1.backward()
        #self-supervised forward and loss
        rot_images, rot_labels = iter(data_loader_rot).next()
        rot_images, rot_labels = rot_images.cuda(), rot_labels.cuda()
        rot_output = model(rot_images, mode='classify')
        loss2 = criterion2(rot_output, rot_labels) * loss_adjust[1]
        loss = loss1 + loss2
        loss.backward()

        optimizer.step()
        epoch_loss += loss1.item()
        metric.add(outputs, labels)
        metric2.update(rot_output,rot_labels)

    epoch_loss = epoch_loss / len(data_loader_seg)
    return epoch_loss , metric.value()[1], metric2.get_accuracy()

def run_val_epoch(epoch, model, criterion, criterion2, metric, data_loader_seg, data_loader_rot):
    epoch_loss = 0.0
    model.eval()
    metric.reset()
    metric2 = accuracy()
    for step, batch_data in enumerate(data_loader_seg):
        with torch.no_grad():
            inputs, labels = batch_data     
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            outputs = model(inputs, mode='full')
            loss1 = criterion(outputs, labels)
            #self-supervised forward and loss
            rot_images, rot_labels = iter(data_loader_rot).next()
            rot_images, rot_labels = Variable(rot_images).cuda(), Variable(rot_labels).cuda()
            rot_output = model(rot_images, mode='classify')
            loss2 = criterion2(rot_output, rot_labels)
            #record metrics
            epoch_loss += loss1.item()
            metric.add(outputs, labels)
            metric2.update(rot_output,rot_labels)
    
    epoch_loss = epoch_loss / len(data_loader_seg)
    return epoch_loss , metric.value()[1], metric2.get_accuracy()

def save_model(model, optimizer, model_path, epoch, val_best_enc, val_best_dec):
    print('saving best model')
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_best_enc': val_best_enc,
        'val_best_dec': val_best_dec}
        
    torch.save(checkpoint, model_path)

    summary_filename = SAVE_PATH+'/erfnet_multihead_summary.txt'
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)
        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("val_best_enc: {0}\n". format(val_best_enc))
        summary_file.write("val_best_dec: {0}\n". format(val_best_dec))
    
def main():
    #Prepare Data
    train_set_seg = dataset_cityscapes(root_dir=DATASET_DIR, mode='train', height=256, width=512)
    train_loader_seg = data.DataLoader(train_set_seg, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_set_seg = dataset_cityscapes(root_dir=DATASET_DIR, mode='val', height=256, width=512)
    val_loader_seg = data.DataLoader(val_set_seg, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    #self-supervised
    train_set_rot = dataset_rot(root_dir=DATASET_DIR, mode='train', dataset='city')
    train_loader_rot = data.DataLoader(train_set_rot, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_set_rot = dataset_rot(root_dir=DATASET_DIR, mode='val',dataset='city')
    val_loader_rot = data.DataLoader(val_set_rot, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print('Batch Size: ',BATCH_SIZE)
    print("Train dataset size:", len(train_set_seg))
    print("Val dataset size:", len(val_set_seg))
    print("Train selfsup dataset size:", len(train_set_seg))
    print("Val selfsup dataset size:", len(val_set_seg))
    images, labels = iter(train_loader_seg).next()
    images2, labels2 = iter(train_loader_rot).next()
    print("Train Image size:", images.size())
    print("Train Label size:", labels.size())
    print("Selfsup Train Image size:", images2.size())
    print("Selfsup Train Label size:", labels2.size())

    #Load Model
    if args.premode == 'none':
        pretrainedEnc = None
        encoderParamsCnt = 0
    elif args.premode == 'imagenet':
        print("Loading encoder pretrained on imagenet classification")
        pretrained = ERFNet_normal(1000,classify=True)
        checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/classification/' + args.pretrain_name)
        print("Loading: ",args.pretrain_name)
        pretrained.load_state_dict(checkpnt['state_dict'])
        pretrainedEnc = pretrained.encoder
        
    #Setup Model
    model = ERFNet(num_classes_dec=20,num_classes_enc=4,encoder=pretrainedEnc).cuda()
    model_path = SAVE_PATH+'/erfnet.pth'
    freeze_model_params = freezeParams(model,57)
    #freeze train
    print('Freeze Optimizer: Adam')
    optimizer = optim.Adam(freeze_model_params, lr=FREEZE_LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    print('Criterion: CrossEntropy')
    class_weights = get_class_weights()
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    metric = IoU(20, ignore_index=19)
    lambda1 = lambda epoch: pow((1-((epoch-1)/NUM_EPOCHS)),0.9)
    lr_updater = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # loss_adjust = [1,0.66]
    print('Training in Freeze mode')
    for epoch in range(FREEZE_NUM_EPOCHS):
        loss_adjust = [(np.cos(epoch/100*np.pi)+1)/2, (np.cos((epoch+75)/100*np.pi)+1)/2]
        epoch_loss, train_miou, self_sup_acc = run_train_epoch(epoch, model, criterion, optimizer, metric, train_loader_seg, train_loader_rot, loss_adjust)
        print(">> FREEZE Epoch: %d | Loss: %2.4f | Train Miou: %2.4f | Self-sup Acc: %2.4f" %(epoch, epoch_loss, train_miou, self_sup_acc))
        val_loss, val_miou, val_self_sup_acc = run_val_epoch(epoch, model, criterion, metric, val_loader_seg, val_loader_rot)
        lr_updater.step(epoch)
        print(">>>> FREEZE Val Epoch: %d | Loss: %2.4f | Val Miou: %2.4f | Self-sup Acc: %2.4f" %(epoch, val_loss, val_miou, val_self_sup_acc ))

    #final train
    print('training full model')
    model_params = []
    model_params.extend(model.parameters())
    for params in model_params:
        params.requires_grad = True
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    criterion2 = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model_params, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    lambda1 = lambda epoch: pow((1-((epoch-1)/NUM_EPOCHS)),0.9)
    lr_updater = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    best_val_miou = 0
    #>>>>>>>>>>>>>>>TRAIN<<<<<<<<<<<<<<<<
    for epoch in range(NUM_EPOCHS):
        loss_adjust = [(np.cos(epoch/200*np.pi)+1)/2, (np.cos((epoch+200)/200*np.pi)+1)/2]
        epoch_loss, train_miou, self_sup_acc = run_train_epoch(epoch, model, criterion, criterion2, optimizer, metric, train_loader_seg, train_loader_rot, loss_adjust)
        print(">> Epoch: %d | Loss: %2.4f | Train Miou: %2.4f | Self-sup Acc: %2.4f" %(epoch, epoch_loss, train_miou, self_sup_acc))
        val_loss, val_miou, val_self_sup_acc = run_val_epoch(epoch, model, criterion, criterion2, metric, val_loader_seg, val_loader_rot)
        lr_updater.step(epoch)
        print(">>>> Val Epoch: %d | Loss: %2.4f | Val Miou: %2.4f | Self-sup Acc: %2.4f" %(epoch, val_loss, val_miou, val_self_sup_acc ))
        
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            save_model(model, optimizer, model_path,epoch, best_val_miou, val_self_sup_acc)


if __name__ == '__main__':
    main()