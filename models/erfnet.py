import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import time

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64)) #0

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.1, 1))  

        self.layers.append(DownsamplerBlock(64,128))#6

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))

    def forward(self, input, predict=False):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        
        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)
        return output

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.l1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(128, num_classes)

    def forward(self, input):
        # print(input.size())
        output = self.pool(input)
        # print(output.size())
        output = torch.flatten(output, 1)
        # print(output.size())
        output = self.l1(output)
        # print(output.size())
        output = self.relu(output)
        output = self.l2(output)
        return output


class ERFNet(nn.Module):
    def __init__(self, num_classes, encoder=None, classify=False, decoder=None):  #use encoder to pass pretrained encoder
        super().__init__()
        self.classify = classify
        if classify:
            self.classifier = Classifier(num_classes)
        else:
            if (decoder == None):
                self.decoder = Decoder(num_classes)
            else:
                self.decoder = decoder

        if (encoder == None):
            self.encoder = Encoder()
        else:
            self.encoder = encoder

    def forward(self, input):
        if self.classify:
            output = self.encoder(input)
            output = self.classifier(output)
            return output
        else:
            output = self.encoder(input)
            output = self.decoder(output)
            return output

if __name__ == "__main__":
    model = ERFNet(20,classify=True).cuda()
    # encoder = model.encoder
    # for param in encoder.parameters():
    #     print(param.data[0].pow(2).sum())
    #     exit()
    model.eval()
    with torch.no_grad():
        input = torch.rand(1, 3, 224, 224).cuda()
        start = time.time()
        output = model(input)
        print(time.time()-start)
        print(output.size())