from models.erfnet import ERFNet
import torch
import utils
from path import get_root_path

ROOT_PATH = get_root_path()
print("Loading encoder pretrained on imagenet cluster")
model = ERFNet(20)
# checkpnt = torch.load(ROOT_PATH+'/Documents/Quad-S-Learning/save/cluster/erfnet_encoder.pth')
# pretrained.load_state_dict(checkpnt['state_dict'])
# pretrained.decoder.output_conv = torch.nn.ConvTranspose2d( 16, 20, 2, stride=2, padding=0, output_padding=0, bias=True)
# output_layer_params = pretrained.decoder.output_conv.parameters()
params = utils.freezeParams(model,57)

for cnt,(name, param) in enumerate(model.named_parameters()):
        print(cnt,' ',param.requires_grad, ' ', name )

#freeze train
        
for param in model.parameters():
        param.requires_grad = True

for cnt,(name, param) in enumerate(model.named_parameters()):
        print(cnt,' ',param.requires_grad, ' ', name )