from models.erfnet import ERFNet
import torch
import utils
import numpy as np
import time
from torchsummary import summary
from path import get_root_path

ROOT_PATH = get_root_path()


model = ERFNet(20).cuda()
# checkpoint = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/ed/erfnet_ed.pth')
checkpoint = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/exp/erfnet_exp_ed_lotto.pth')
model.load_state_dict(checkpoint['state_dict'])

# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
params = 0
for p in model.parameters():
    if len(p.size()) > 1:
        params += int(np.prod(p.size())*0.2)
    else:
        params += np.prod(p.size())
    

# summary(model, (3, 512, 1024))
print(params)
exit()
inputs = []
for x in range(10):
    input = torch.rand(1, 3, 512, 1024).cuda()
    inputs.append(input)
model.eval()
with torch.no_grad():
    start = time.time()
    for x in range(10):
        outputs = model(inputs[x])
    end = time.time()
    elapse = (end-start)*100
print("elapse time: ",elapse )

