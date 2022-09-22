from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from model import FasterRCNNVGG16_PRUNE
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
import resource
import torch.nn as nn
import torch
import numpy as np
import pickle
from torch.autograd import Variable

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

# *********************************测试模型的代码****************************************
def test(model):
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       # pin_memory=True
                                       )
    model.eval()
    correct = 0
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    test_num=opt.test_num
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(test_dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = model.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result

# ************************************加载剪枝前的稀疏化模型********************************************
opt.student = False
tea_cfg=opt.tea_cfg
faster_rcnn = FasterRCNNVGG16(tea_cfg=tea_cfg)
faster_rcnn.cuda()
print("======================剪枝前的faster_rcnn==========================")
print(faster_rcnn)
if opt.load_path:
    if os.path.isfile(opt.load_path):
        print("=> loading checkpoint '{}'".format(opt.load_path))
        checkpoint = torch.load(opt.load_path)
        faster_rcnn.load_state_dict(checkpoint['model'])
        # print("教师模型的")
        # print(checkpoint['model'])
    else:
        print("=> no checkpoint found at '{}'".format(opt.load_path))


# ***********************************计算重要性********************************************
cfg = opt.cfg

cfg_mask = []
layer_id = 0
for m in faster_rcnn.modules():
    if layer_id == 17:
        break
    if isinstance(m, nn.Conv2d):
        out_channels = m.weight.data.shape[0]
        if out_channels == cfg[layer_id]:
            cfg_mask.append(torch.ones(out_channels))
            layer_id += 1
            continue
        weight_copy = m.weight.data.abs().clone()
        weight_copy = weight_copy.cpu().numpy()
        L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
        arg_max = np.argsort(L1_norm)
        arg_max_rev = arg_max[::-1][:cfg[layer_id]]
        assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
        mask = torch.zeros(out_channels)
        mask[arg_max_rev.tolist()] = 1
        cfg_mask.append(mask)
        layer_id += 1
    elif isinstance(m, nn.MaxPool2d):
        layer_id += 1


# ***************************************中间结果的处理*******************************************
# teacher_acc = test(faster_rcnn)
# print('大模型的精确度: {}.'.format(str(teacher_acc)))
opt.student = True
opt.prune = True
newmodel = FasterRCNNVGG16_PRUNE(cfg=cfg)
newmodel.cuda()



# 保存cfg_mask的信息
savepath = os.path.join("checkpoints", "cfg_mask.txt")
fileHandle = open(savepath, 'wb')
testList = cfg_mask
pickle.dump(testList, fileHandle)
fileHandle.close()

# 读取保存的cfg_mask的信息
# fileHandle = open(savepath, 'rb')
# testList = pickle.load(fileHandle)
# fileHandle.close()
# print(testList)


# ******************************开始真正的剪枝，并且将保留的filter的权重赋给小模型***********************************
# Make real prune
# print(cfg_mask)  # cfg_mask是正确的
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
# modules()返回一个包含 当前模型 所有模块的迭代器。
layer_num_index = 0
for [m0, m1] in zip(faster_rcnn.modules(), newmodel.modules()):
    layer_num_index += 1
    if isinstance(m0, nn.BatchNorm2d):
        # np.argwhere：返回非0的数组元组的索引，其中a是要索引数组的条件
        # np.squeeze：返回值与原tensor共享内存，修改返回值中元素值对原tensor也有影响
        # 将没有被剪枝的值付给新模型的weight
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))  # 没有被剪枝的filter的索引[  2   5   6   9  10  11  15  20  22  23  32
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        # 将原模型未剪枝的filter的值赋给新模型
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1  # 开始看下一个卷积层
        start_mask = end_mask
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC#到时候修改此处匹配自己的
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        if layer_num_index == 35: # RPN的score层：Conv2d(512, 18, kernel_size=(1, 1), stride=(1, 1))
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))  # 上一层的mask，即该层输入的filter的索引值
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, 18))  # In shape: 3, Out shape 14.
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[:, :, :, :].clone()
            m1.weight.data = w1.clone()

            m1.bias.data = m0.bias.data.clone()
        elif layer_num_index == 36: # RPND的loc层：Conv2d(512, 36, kernel_size=(1, 1), stride=(1, 1))
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))  # 上一层的mask，即该层输入的filter的索引值
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, 36))  # In shape: 3, Out shape 14.

            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[:, :, :, :].clone()
            m1.weight.data = w1.clone()

            m1.bias.data = m0.bias.data.clone()
        else:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))  # 上一层的mask，即该层输入的filter的索引值
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))  # 下一层的mask，该层输出的filter的索引值
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))  # In shape: 3, Out shape 14.
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()

            w2 = m0.bias.data[idx1.tolist()].clone()
            m1.bias.data = w2.clone()

            layer_id_in_cfg += 1  # 开始看下一个卷积层
            start_mask = end_mask
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC#到时候修改此处匹配自己的
                end_mask = cfg_mask[layer_id_in_cfg]

    elif isinstance(m0, nn.BatchNorm1d):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()

    elif isinstance(m0, nn.Linear):
        t = 2816
        if layer_num_index == 39:
            w1 = m0.weight.data[0:t, 0:opt.cfg[16] * 7 * 7].clone()
            w2 = m0.bias.data[0:t].clone()
        elif layer_num_index == 41:
            w1 = m0.weight.data[0:t, 0:t].clone()
            w2 = m0.bias.data[0:t].clone()
        elif layer_num_index == 43:
            w1 = m0.weight.data[0:84, 0:t].clone()
            w2 = m0.bias.data[0:84].clone()
        elif layer_num_index == 44:
            w1 = m0.weight.data[0:21, 0:t].clone()
            w2 = m0.bias.data[0:21].clone()
        m1.weight.data = w1
        m1.bias.data = w2


# *************************************保存新模型的各个部位********************************************
torch.save({'state_dict': newmodel.head.state_dict()}, os.path.join("checkpoints", 'head.pth'))
torch.save({'rpn':newmodel.rpn.state_dict()}, os.path.join("checkpoints", 'rpn.pth'))
torch.save({'state_dict': newmodel.extractor.state_dict()}, os.path.join("checkpoints", 'extractor.pth'))
torch.save(newmodel.state_dict(), os.path.join("checkpoints", 'newmodel.pth'))


print("============================剪枝后的newmodel==============================")
checkpoint = torch.load(os.path.join("checkpoints", 'newmodel.pth'))
newmodel.load_state_dict(checkpoint)
# print("学生模型的")
# print(checkpoint)
print(newmodel)
model = newmodel
# student_acc = test(model)
# print('小模型的精确度: {}.'.format(str(student_acc)))

teacher_num_parameters = sum([param.nelement() for param in faster_rcnn.parameters()])
student_num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join("checkpoints", "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of teacher parameters: \n"+str(teacher_num_parameters)+"\n")
    fp.write("Number of student parameters: \n" + str(student_num_parameters) + "\n")
    # fp.write("Teacher Test accuracy: \n" + str(teacher_acc)+"\n")
    # fp.write("Student Test accuracy: \n" + str(student_acc)+"\n")
    fp.write("剪枝后的newmodel: \n" + str(newmodel) + "\n")
    fp.write("cfg_mask: \n"+str(cfg_mask))

