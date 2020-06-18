from models.erfnet import ERFNet as ERFNet
from models.erfnet_jigsaw import ERFNet as ERFNet_jig
from models.erfnet import DownsamplerBlock
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

ROOT_PATH = get_root_path()
DATASET_DIR = ROOT_PATH + "/Documents/Dataset/"
SAVE_PATH = ROOT_PATH + '/Documents/Quad-S-Learning/save/exp/'
_EPS = 1e-9

parser = argparse.ArgumentParser()
parser.add_argument( "--resume", action='store_true')
parser.add_argument( "--epochs",         type=int,   default=150)
parser.add_argument( "--freeze-epochs",         type=int,   default=0)
parser.add_argument( "--batch-size",     type=int,   default=6)
parser.add_argument( "--learning-rate",  type=float, default=0.0005)
parser.add_argument( "--freeze-learning-rate",  type=float, default=0.0005)
parser.add_argument( "--workers",        type=int,   default=12)
parser.add_argument( "--encoder",           choices=['none', 'class','rot'], default='rot')
parser.add_argument( "--dataset",        choices=['cityscapes','camvid','voc'],  default='cityscapes')
parser.add_argument( "--height",         type=int, default=512)
parser.add_argument( "--width",          type=int, default=1024)
parser.add_argument( "--pretrain-enc-name",  type=str, default='erfnet_encoder_rot_city_waug.pth')
parser.add_argument( "--name",           type=str, default='erfnet_explotto.pth')
args = parser.parse_args()

def zero_weights(layer):
    w = layer.weight.data.clone()
    w_shape = w.size()
    w = torch.abs(w)
    w = w.flatten()
    indexes = w.argsort() #from smallest to largest
    i = len(indexes)//2
    mask = torch.zeros(w.size(),dtype=torch.bool)
    mask[indexes[:i]] = True
    mask = mask.reshape(w_shape)
    layer.weight.data[mask] = 0.0
    return mask

def zero_param(param):
    w = param.data.clone()
    w_shape = w.size()
    w = torch.abs(w)
    w = w.flatten()
    indexes = w.argsort() #from smallest to largest
    i = len(indexes)//2
    mask = torch.zeros(w.size(),dtype=torch.bool)
    mask[indexes[:i]] = True
    mask = mask.reshape(w_shape)
    
    # param.data[mask] = 0.0
    del w
    return mask

def lotto(model):
    print('zeroing weights')
    mask_list = [] #list of mask True should stay 0.
    i=0
    for block in model.children():
        if isinstance(block, DownsamplerBlock):#get initial block
            for layer in block.children():
                if isinstance(layer, nn.Conv2d):
                    # mask_list.append(zero_weights(layer))
                    i+=1
                elif isinstance(layer,nn.BatchNorm2d):
                    layer.reset_parameters()
        elif isinstance(block, nn.ModuleList):#main meat of erfnet
            for subblock in block.children():#non-bt-1d and downsamplers
                for layer in subblock.children():#actual conv2d and batchnorm
                    if isinstance(layer, nn.Conv2d):
                        # mask_list.append(zero_weights(layer))
                        i+=1
                    elif isinstance(layer,nn.BatchNorm2d):
                        layer.reset_parameters()

def get_class_weights():
    if args.dataset == 'cityscapes':
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
    elif args.dataset == 'camvid':
        class_weights = np.ones(12)
        class_weights[0] = 5.79203481	
        class_weights[1] = 4.44028773	
        class_weights[2] = 34.02166463	
        class_weights[3] = 3.44690044	
        class_weights[4] = 15.91194352	
        class_weights[5] = 9.02023585	
        class_weights[6] = 32.01377376	
        class_weights[7] = 32.47892445	
        class_weights[8] = 13.20714089	
        class_weights[9] = 38.38765298	
        class_weights[10] = 44.13450551	
        class_weights[11] = 17.30636391

    class_weights = torch.from_numpy(class_weights).float()
    return class_weights

def freezeParams(model,cnt):
    model_params = []
    model_params.extend(model.parameters())
    for i, param in enumerate(model_params):
        if i > cnt:
            param.requires_grad = False
    
    return model_params
                                 
def run_train_epoch(epoch, model, criterion, optimizer, metric, data_loader, mask_list):
    epoch_loss = 0.0
    model.train()
    metric.reset()
    for step, batch_data in enumerate(data_loader):
        inputs, labels = batch_data     
        # inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        #keep weights at 0 so set their grad to 0.
        i = 0
        for param in model.encoder.parameters():
            if len(param.size()) > 1 :                
                param.grad.data[mask_list[i]] = 0.0
                i += 1
        optimizer.step()
        epoch_loss += loss.item()
        metric.add(outputs, labels)

    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss , metric.value()[1]

def run_val_epoch(epoch, model, criterion, metric, data_loader):
    epoch_loss = 0.0
    model.eval()
    metric.reset()
    for step, batch_data in enumerate(data_loader):
        with torch.no_grad():
            inputs, labels = batch_data     
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            metric.add(outputs, labels)
    
    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss , metric.value()[1]

def save_model(model, optimizer, model_path, epoch, val_best_miou):
    checkpoint = {
        'epoch': epoch,
        'val_best_miou': val_best_miou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
        
    torch.save(checkpoint, model_path)

    summary_filename = SAVE_PATH + args.name + '_summary.txt'
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)
        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("val_best_miou: {0}\n". format(val_best_miou))

def main():
    #Prepare Data
    train_set = dataset(root_dir=DATASET_DIR, mode='train', height=args.height, width=args.width)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_set = dataset(root_dir=DATASET_DIR, mode='val', height=args.height, width=args.width)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
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
    #Load Model encoder
    if not args.resume:
        if args.encoder == 'class':
            print("Loading encoder pretrained on imagenet classification")
            pretrained = ERFNet(1000,classify=True)
            checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/classification/' + args.pretrain_enc_name)
            print("Loading encoder: ", args.pretrain_enc_name)
            pretrained.load_state_dict(checkpnt['state_dict'])
            pretrainedEnc = pretrained.encoder
        elif args.encoder == 'rot':
            print("Loading encoder pretrained on rotations")
            pretrained = ERFNet(4,classify=True)
            checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/rotation/' + args.pretrain_enc_name)
            print("Loading encoder: ", args.pretrain_enc_name)
            pretrained.load_state_dict(checkpnt['state_dict'])
            pretrainedEnc = pretrained.encoder

        mask_list = []
        for param in pretrainedEnc.parameters():
            print(param.size())
            if len(param.size()) > 1 :
                mask = zero_param(param).cuda()
                mask_list.append(mask)
                param.data[mask] = 0.0
                # print(param.data)
        # lotto(pretrainedEnc)
        # mask_list = mask_list.cuda()
    else:
        pretrainedEnc = None
    
    # exit()
    #Setup Model
    model = ERFNet(num_classes=num_classes, encoder=pretrainedEnc, decoder=None).cuda()
    # for name, param in model.named_parameters():
    #     print(param.data)
    #     print(name)
    #     exit()
        # if param.size()[0] == 13:
        #     print(param.data)
        #     print(name)
    #         exit()
    model_path = SAVE_PATH + args.name
    class_weights = get_class_weights()
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    print('Criterion: CrossEntropy')
    metric = IoU(num_classes, ignore_index=ignore_index)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    print('optimizer: Adam')
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.epochs)),0.9)
    lr_updater = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    print('LR-Updater: Poly')
    if args.resume:
        print('resuming from checkpoint')
        checkpoint = torch.load(SAVE_PATH + args.name)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        best_val_miou = checkpoint['val_best_miou']
        print('start epoch: ', start_epoch, 'best val miou:', best_val_miou)
    else:
        best_val_miou = 0
        start_epoch = 0
    print('Training.....')
    #>>>>>>>>>>>>>>>TRAIN<<<<<<<<<<<<<<<<
    for epoch in range(start_epoch,args.epochs):
        print('Train LR: ', lr_updater.get_lr()[0])
        epoch_loss, train_miou = run_train_epoch(epoch, model, criterion, optimizer, metric, train_loader,mask_list)
        print(">> Epoch: %d | Loss: %2.4f | Train Miou: %2.4f" %(epoch, epoch_loss, train_miou))
        val_loss, val_miou = run_val_epoch(epoch, model, criterion, metric, val_loader)
        lr_updater.step(epoch)
        print(">>>> Val Epoch: %d | Loss: %2.4f | Val Miou: %2.4f" %(epoch, val_loss, val_miou ))
        
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            print('saving best model...')
            save_model(model, optimizer, model_path,epoch, best_val_miou)
    print(best_val_miou)

if __name__ == '__main__':
    if args.dataset == 'cityscapes':
        from data import Cityscapes as dataset
    elif args.dataset == 'camvid':
        from data import Camvid as dataset
    main()