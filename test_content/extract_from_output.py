import re
import matplotlib.pyplot as plt
import numpy as np

x_list = range(150)
y_list_bad = []
f = open("erfNet_train_cityscapes_fsjig_citynoaug.o", "r")
for line in f:
  x = re.match(">>>> \[Epoch: \d+\] Avg. loss: \d+\.\d+ \| Mean IoU: (0\.\d+)",line)
  if x is not None:
      val = float(x[1])*100
      y_list_bad.append(val)

y_list_good = []
f = open("erfNet_train_cityscapes_classwaug.o", "r")
for line in f:
  x = re.match(">>>> \[Epoch: \d+\] Avg. loss: \d+\.\d+ \| Mean IoU: (0\.\d+)",line)
  if x is not None:
      val = float(x[1])*100
      y_list_good.append(val)

y_list_ok = []
f = open("erfNet_train_cityscapes_simclr_city.o", "r")
for line in f:
  x = re.match(">>>> \[Epoch: \d+\] Avg. loss: \d+\.\d+ \| Mean IoU: (0\.\d+)",line)
  if x is not None:
      val = float(x[1])*100
      y_list_ok.append(val)

plt.plot(y_list_bad,'r',label='city fsjigsaw')
plt.plot(y_list_good,'b',label='ImageNet class')
plt.plot(y_list_ok,'g',label='city SimCLR')
plt.xticks(np.arange(0, 165, step=15))
plt.legend(loc='lower right')
plt.ylabel('Val mIoU')
plt.xlabel('Epoch')
plt.title('Initial weights effect on performance.')
plt.show()