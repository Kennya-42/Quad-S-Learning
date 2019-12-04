from torch.autograd import Variable
import torch


class Test():
    
    def __init__(self, model, data_loader, criterion, metric):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric

    def run_epoch(self, iteration_loss=False):
        epoch_loss = 0.0
        self.metric.reset()
        self.model.eval()
        with torch.no_grad():
            for step, batch_data in enumerate(self.data_loader):
                # Get the inputs and labels
                inputs, labels = batch_data

                # Wrap them in a Varaible
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.cuda()
                labels = labels.cuda()

                # Forward propagation
                outputs = self.model(inputs)

                # Loss computation
                loss = self.criterion(outputs, labels)

                # Keep track of loss for current epoch
                epoch_loss += loss.item()

                # Keep track of evaluation the metric
                self.metric.add(outputs.data, labels.data)

                if iteration_loss:
                    print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()
