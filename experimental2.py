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
parser.add_argument( "--encoder2",           choices=['none', 'class','colorize','otsu','jigsaw'], default='jigsaw')
parser.add_argument( "--dataset",        choices=['cityscapes','camvid','voc'],  default='cityscapes')
parser.add_argument( "--height",         type=int, default=512)
parser.add_argument( "--width",          type=int, default=1024)
parser.add_argument( "--pretrain-enc-name",  type=str, default='erfnet_encoder_rot_city_waug.pth')
parser.add_argument( "--pretrain-enc-name2",  type=str, default='erfnet_encoder_jigsaw_city_waug.pth')
parser.add_argument( "--name",           type=str, default='erfnet_exp2rotandjigsaw.pth')
args = parser.parse_args()

def init_low_mag_layer_conv(m, w1, w2, b1, b2, percent_save= 0.9):
    w3 = torch.empty(w1.size())
    b3 = torch.empty(b1.size())
    m1 = w1.pow(2).sqrt().sum([1,2,3])
    m2 = w2.pow(2).sqrt().sum([1,2,3])
    i1 = m1.argsort(0,descending=True)
    i2 = m2.argsort(0,descending=True)
    num_transfer = int(len(i1)*percent_save) #retain a percent of original weights.
    for i in range(len(i1)):
        if i < num_transfer:
            w3[i] = w1[i1[i]]
            b3[i] = b1[i1[i]]
        else:
            w3[i] = w2[i2[i]]
            b3[i] = b2[i2[i]]
    m.weight.data = w3
    m.bias.data = b3

def mix_featuremaps(model,transplant_model):
    print('mixing weights')
    model1_weights = []
    model2_weights = []
    model1_bias = []
    model2_bias = []
    #
    #model1 get weights
    for block in model.children():
        if isinstance(block, DownsamplerBlock):#get initial block
            for layer in block.children():
                if isinstance(layer, nn.Conv2d):
                    model1_weights.append(layer.weight.data)
                    model1_bias.append(layer.bias.data)
                
        elif isinstance(block, nn.ModuleList):#main meat of erfnet
            for subblock in block.children():#non-bt-1d and downsamplers
                for layer in subblock.children():#actual conv2d and batchnorm
                    if isinstance(layer, nn.Conv2d):
                        model1_weights.append(layer.weight.data)
                        model1_bias.append(layer.bias.data)

    #
    #model2 get weights
    for block in model.children():
        if isinstance(block, DownsamplerBlock):#get initial block
            for layer in block.children():
                if isinstance(layer, nn.Conv2d):
                    model2_weights.append(layer.weight.data)
                    model2_bias.append(layer.bias.data)
                
        elif isinstance(block, nn.ModuleList):#main meat of erfnet
            for subblock in block.children():#non-bt-1d and downsamplers
                for layer in subblock.children():#actual conv2d and batchnorm
                    if isinstance(layer, nn.Conv2d):
                        model2_weights.append(layer.weight.data)
                        model2_bias.append(layer.bias.data)
    
    #
    #Load weights into base model, reset batchnorm
    i=0
    for block in model.children():
        if isinstance(block, DownsamplerBlock):#get initial block
            for layer in block.children():
                if isinstance(layer, nn.Conv2d):
                    init_low_mag_layer_conv(layer,model1_weights[i],model2_weights[i],model1_bias[i],model2_bias[i])
                    i+=1
                elif isinstance(layer,nn.BatchNorm2d):
                    layer.reset_parameters()
        elif isinstance(block, nn.ModuleList):#main meat of erfnet
            for subblock in block.children():#non-bt-1d and downsamplers
                for layer in subblock.children():#actual conv2d and batchnorm
                    if isinstance(layer, nn.Conv2d):
                        init_low_mag_layer_conv(layer,model1_weights[i],model2_weights[i],model1_bias[i],model2_bias[i])
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
                                 
def run_train_epoch(epoch, model, criterion, optimizer, metric, data_loader):
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

        if args.encoder2 == 'class':
            print("Loading encoder pretrained on imagenet classification")
            pretrained = ERFNet(1000,classify=True)
            checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/classification/' + args.pretrain_enc_name2)
            print("Loading encoder: ", args.pretrain_enc_name)
            pretrained.load_state_dict(checkpnt['state_dict'])
            pretrainedEnc2 = pretrained.encoder
        elif args.encoder2 == 'rot':
            print("Loading encoder pretrained on rotations")
            pretrained = ERFNet(4,classify=True)
            checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/rotation/' + args.pretrain_enc_name2)
            print("Loading encoder: ", args.pretrain_enc_name)
            pretrained.load_state_dict(checkpnt['state_dict'])
            pretrainedEnc2 = pretrained.encoder
        elif args.encoder2 == 'otsu':
            print("Loading encoder2 pretrained on otsu clustering")
            pretrained = ERFNet(4, classify=False)
            checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/cluster/otsu/' + args.pretrain_enc_name2)
            print("Loading encoder2: ",args.pretrain_enc_name2)
            pretrained.load_state_dict(checkpnt['state_dict'])
            pretrainedEnc2 = pretrained.encoder
        elif args.encoder2 == 'jigsaw':
            print("Loading encoder2 pretrained on jigsaw")
            pretrained = ERFNet_jig(num_classes=1000)
            checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/jigsaw/' + args.pretrain_enc_name2)
            print("Loading: ",args.pretrain_enc_name2)
            pretrained.load_state_dict(checkpnt['state_dict'])
            pretrainedEnc2 = pretrained.encoder

        mix_featuremaps(pretrainedEnc, pretrainedEnc2)
    else:
        pretrainedEnc = None

    # exit()
    #Setup Model
    model = ERFNet(num_classes=num_classes, encoder=pretrainedEnc, decoder=None).cuda()
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
        epoch_loss, train_miou = run_train_epoch(epoch, model, criterion, optimizer, metric, train_loader)
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