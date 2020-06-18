import os
from PIL import Image, ImageOps, ImageCms
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import math
from torch.optim import SGD, Adam, lr_scheduler
from erfnet import ERFNet

num_epochs = 150
model = ERFNet(20)
optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
lambda1 = lambda epoch: pow((1-((epoch-1)/num_epochs)),0.9)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
y_list = []
for i in range(num_epochs):
        scheduler.step(i)
        curr_lr = scheduler.get_lr()
        y_list.append(curr_lr)
plt.plot(y_list,)
plt.xticks(np.arange(0, 160, 10))
plt.xlabel("Epochs")
plt.ylabel("Learning rate")
plt.show()
print(np.max(y_list))
print(np.min(y_list))
#0.00001026632166