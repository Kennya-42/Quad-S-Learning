from torch.autograd import Variable
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import utils
from collections import OrderedDict
class Train():
    def __init__(self, model, data_loader, optim, criterion, metric,lr_updater):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.lr_updater = lr_updater

    def run_epoch(self, epoch, iteration_loss=False):
        epoch_loss = 0.0
        self.metric.reset()
        self.model.train()
        for step, batch_data in enumerate(self.data_loader):
            inputs, labels = batch_data     
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            self.lr_updater(self.optim, step, epoch)
            #Forward Propagation
            outputs = self.model(inputs)
            # Loss computation
            loss = self.criterion(outputs, labels)
            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # Keep track of loss for current epoch
            epoch_loss += loss.item()
            # Keep track of the evaluation metric
            self.metric.add(outputs.data, labels.data)
            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()
