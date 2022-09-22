# code from ruotian luo
# https://github.com/ruotianluo/pytorch-faster-rcnn
import torch
from torch.utils.model_zoo import load_url
from torchvision import models

# sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth")
# sd['classifier.0.weight'] = sd['classifier.1.weight']
# sd['classifier.0.bias'] = sd['classifier.1.bias']
# del sd['classifier.1.weight']
# del sd['classifier.1.bias']
#
# sd['classifier.3.weight'] = sd['classifier.4.weight']
# sd['classifier.3.bias'] = sd['classifier.4.bias']
# del sd['classifier.4.weight']
# del sd['classifier.4.bias']
#
# import  os
# # speicify the path to save
# if not os.path.exists('checkpoints'):
#     os.makedirs('checkpoints')
# torch.save(sd, "checkpoints/vgg16_caffe.pth")

list_vgg16 = ['features.0.weight', 'features.0.bias',
'features.2.weight', 'features.2.bias',
'features.5.weight', 'features.5.bias',
'features.7.weight', 'features.7.bias',
'features.10.weight', 'features.10.bias',
'features.12.weight', 'features.12.bias',
'features.14.weight', 'features.14.bias',
'features.17.weight', 'features.17.bias',
'features.19.weight', 'features.19.bias',
'features.21.weight', 'features.21.bias',
'features.24.weight', 'features.24.bias',
'features.26.weight', 'features.26.bias',
'features.28.weight', 'features.28.bias']

list_vgg16_bn = ['features.0.weight', 'features.0.bias',
'features.1.weight', 'features.1.bias', 'features.1.running_mean', 'features.1.running_var',
'features.3.weight', 'features.3.bias',
'features.4.weight', 'features.4.bias', 'features.4.running_mean', 'features.4.running_var',
'features.7.weight', 'features.7.bias',
'features.8.weight', 'features.8.bias', 'features.8.running_mean', 'features.8.running_var',
'features.10.weight', 'features.10.bias',
'features.11.weight', 'features.11.bias', 'features.11.running_mean', 'features.11.running_var',
'features.14.weight', 'features.14.bias',
'features.15.weight', 'features.15.bias', 'features.15.running_mean', 'features.15.running_var',
'features.17.weight', 'features.17.bias',
'features.18.weight', 'features.18.bias', 'features.18.running_mean', 'features.18.running_var',
'features.20.weight', 'features.20.bias',
'features.21.weight', 'features.21.bias', 'features.21.running_mean', 'features.21.running_var',
'features.24.weight', 'features.24.bias',
'features.25.weight', 'features.25.bias', 'features.25.running_mean', 'features.25.running_var',
'features.27.weight', 'features.27.bias',
'features.28.weight', 'features.28.bias', 'features.28.running_mean', 'features.28.running_var',
'features.30.weight', 'features.30.bias',
'features.31.weight', 'features.31.bias', 'features.31.running_mean', 'features.31.running_var',
'features.34.weight', 'features.34.bias',
'features.35.weight', 'features.35.bias', 'features.35.running_mean', 'features.35.running_var',
'features.37.weight', 'features.37.bias',
'features.38.weight', 'features.38.bias', 'features.38.running_mean', 'features.38.running_var',
'features.40.weight', 'features.40.bias',
'features.41.weight', 'features.41.bias', 'features.41.running_mean', 'features.41.running_var']

list_vgg16_bn_conv = ['features.0.weight', 'features.0.bias',
'features.3.weight', 'features.3.bias',
'features.7.weight', 'features.7.bias',
'features.10.weight', 'features.10.bias',
'features.14.weight', 'features.14.bias',
'features.17.weight', 'features.17.bias',
'features.20.weight', 'features.20.bias',
'features.24.weight', 'features.24.bias',
'features.27.weight', 'features.27.bias',
'features.30.weight', 'features.30.bias',
'features.34.weight', 'features.34.bias',
'features.37.weight', 'features.37.bias',
'features.40.weight', 'features.40.bias']

list_vgg16_bn_bn = [
'features.1.weight', 'features.1.bias', 'features.1.running_mean', 'features.1.running_var',
'features.4.weight', 'features.4.bias', 'features.4.running_mean', 'features.4.running_var',
'features.8.weight', 'features.8.bias', 'features.8.running_mean', 'features.8.running_var',
'features.11.weight', 'features.11.bias', 'features.11.running_mean', 'features.11.running_var',
'features.15.weight', 'features.15.bias', 'features.15.running_mean', 'features.15.running_var',
'features.18.weight', 'features.18.bias', 'features.18.running_mean', 'features.18.running_var',
'features.21.weight', 'features.21.bias', 'features.21.running_mean', 'features.21.running_var',
'features.25.weight', 'features.25.bias', 'features.25.running_mean', 'features.25.running_var',
'features.28.weight', 'features.28.bias', 'features.28.running_mean', 'features.28.running_var',
'features.31.weight', 'features.31.bias', 'features.31.running_mean', 'features.31.running_var',
'features.35.weight', 'features.35.bias', 'features.35.running_mean', 'features.35.running_var',
'features.38.weight', 'features.38.bias', 'features.38.running_mean', 'features.38.running_var',
'features.41.weight', 'features.41.bias', 'features.41.running_mean', 'features.41.running_var']


model_vgg16 = torch.load("../pretrained_model/vgg16-397923af.pth", map_location='cpu')
model_vgg16_bn = torch.load("../pretrained_model/vgg16_bn-6c64b313.pth", map_location='cpu')
for i in range(len(list_vgg16)):
    model_vgg16_bn[list_vgg16_bn_conv[i]] = model_vgg16.pop(list_vgg16[i])
torch.save(model_vgg16_bn, "../pretrained_model/vgg16_bn-caffe.pth")
# sd = torch.load("../pretrained_model/fasterrcnn_11202239_0.6672908711125728", map_location='cpu')
# sd["model"]['head.score.weight'] = torch.nn.init.normal_(torch.rand(21,4096))
# sd["model"]['head.score.bias'] = torch.nn.init.normal_(torch.rand(21))
# sd["model"]['head.cls_loc.weight'] = torch.nn.init.normal_(torch.rand(84, 4096))
# sd["model"]['head.cls_loc.bias'] = torch.nn.init.normal_(torch.rand(84))


# sd["model"]['features.1.weight'] = torch.nn.init.normal_(torch.rand(64))


# import  os
# # speicify the path to save
# if not os.path.exists('checkpoints'):
#     os.makedirs('checkpoints')
# torch.save(sd, "checkpoints/vgg16_caffe.pth")