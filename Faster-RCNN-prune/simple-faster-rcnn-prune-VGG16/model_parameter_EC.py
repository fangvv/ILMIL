from model import FasterRCNNVGG16
from model import FasterRCNNVGG16_PRUNE
from utils.config import opt
from compute_flops_EC import print_model_param_nums, count_model_param_flops
import torch
from torchstat import stat
from torch.autograd import Variable


# *************************************** 原始模型 ****************************************
opt.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
opt.student = False
faster_rcnn = FasterRCNNVGG16()
# print(faster_rcnn)
faster_rcnn.cuda()
H = torch.tensor(600, device='cuda')
W = torch.tensor(800, device='cuda')
print_model_param_nums(faster_rcnn)
count_model_param_flops(faster_rcnn, H, W)