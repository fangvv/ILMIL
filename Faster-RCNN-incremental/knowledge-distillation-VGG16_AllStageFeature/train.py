#!/usr/bin/env python
# -*- coding:utf8 -*-
from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
#import cupy as cp
import os
import numpy as np
import ipdb
import matplotlib
from tqdm import tqdm
import torch as t
import cv2
import resource

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize, Transform, TestDataset_all
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from PIL import Image
from matplotlib import pyplot as plt
from data.util import read_image
from data import util
from utils.utils import *
from torchstat import stat
import argparse
# from uitils import *
# 更改gpu使用的核心
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# 使用作者的模型训练

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')
VOC_BBOX_LABEL_NAMES = opt.VOC_BBOX_LABEL_NAMES

def train(**kwargs):
    opt._parse(kwargs)
    print('load data')

    dataset = Dataset(opt)
    dataloader = data_.DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False,
                                       # pin_memory=True
                                       )
    testset_all = TestDataset_all(opt, 'test')
    test_all_dataloader = data_.DataLoader(testset_all,
                                           batch_size=1,
                                           num_workers=opt.test_num_workers,
                                           shuffle=False,
                                           pin_memory=True
                                           )

    tsf = Transform(opt.min_size, opt.max_size)
    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    print('model construct completed')
    #stat(faster_rcnn, (3, 600, 1000))
    # 加载训练过的模型，在config配置路径就可以了
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    # 提取蒸馏知识所需要的软标签
    # if opt.is_distillation == True:
    #     opt.predict_socre = 0.3
    #     for ii, (imgs, sizes, gt_bboxes_, gt_labels_, scale, id_) in tqdm(enumerate(dataloader)):
    #         if len(gt_bboxes_) == 0:
    #             continue
    #         sizes = [sizes[0][0].item(), sizes[1][0].item()]
    #         pred_bboxes_, pred_labels_, pred_scores_, features_, stage_features4_ = trainer.faster_rcnn.predict(imgs, [sizes])
    #
    #         img_file = os.path.join(opt.voc_data_dir, 'JPEGImages', id_[0] + '.jpg')
    #         ori_img = read_image(img_file, color=True)
    #         img, pred_bboxes_, pred_labels_, scale_ = tsf((ori_img, pred_bboxes_[0], pred_labels_[0]))
    #
    #         #去除软标签和真值标签重叠过多的部分，去除错误的软标签
    #         pred_bboxes_, pred_labels_, pred_scores_ = py_cpu_nms(
    #             gt_bboxes_[0], gt_labels_[0], pred_bboxes_, pred_labels_, pred_scores_[0])
    #
    #         #存储软标签，这样存储不会使得GPU占用过多
    #         np.save('label/' + str(id_[0]) + '.npy', pred_labels_.cpu())
    #         np.save('bbox/' + str(id_[0]) + '.npy', pred_bboxes_.cpu())
    #         np.save('feature/' + str(id_[0]) + '.npy', features_.cpu())
    #         #np.save('stage_features1/' + str(id_[0]) + '.npy', stage_features1_.cpu())
    #         #np.save('stage_features2/' + str(id_[0]) + '.npy', stage_features2_.cpu())
    #         #np.save('stage_features3/' + str(id_[0]) + '.npy', stage_features3_.cpu())
    #         np.save('stage_features4/' + str(id_[0]) + '.npy', stage_features4_.cpu())
    #         np.save('score/' + str(id_[0]) + '.npy', pred_scores_.cpu())
    #
    #     opt.predict_socre = 0.05
    # t.cuda.empty_cache()

    # # visdom 显示所有类别标签名
    # trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr

    for epoch in range(opt.epoch):
        print('epoch=%d' % epoch)

        # 重置混淆矩阵
        trainer.reset_meters()
        # tqdm可以在长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)，
        # 是一个快速、扩展性强
        for ii, (img, sizes, bbox_, label_, scale, id_) in tqdm(enumerate(dataloader)):
            if len(bbox_) == 0:
                continue
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            # 训练的就这一步 下面的都是打印的信息
            # 转化成pytorch能够计算的格式，转tensor格式
            if opt.is_distillation == True:
                # 读取软标签
                teacher_pred_labels = np.load('label/' + str(id_[0]) + '.npy')
                teacher_pred_bboxes = np.load('bbox/' + str(id_[0]) + '.npy')
                teacher_pred_features_ = np.load('feature/' + str(id_[0]) + '.npy')
                # teacher_pred_stage_features1_ = np.load('stage_features1/' + str(id_[0]) + '.npy')
                # teacher_pred_stage_features2_ = np.load('stage_features2/' + str(id_[0]) + '.npy')
                # teacher_pred_stage_features3_ = np.load('stage_features3/' + str(id_[0]) + '.npy')
                teacher_pred_stage_features4_ = np.load('stage_features4/' + str(id_[0]) + '.npy')
                teacher_pred_scores = np.load('score/' + str(id_[0]) + '.npy')
                # 格式转换
                teacher_pred_bboxes = teacher_pred_bboxes.astype(np.float32)
                teacher_pred_labels = teacher_pred_labels.astype(np.int32)
                teacher_pred_scores = teacher_pred_scores.astype(np.float32)
                # 转成pytorch格式
                teacher_pred_bboxes_ = at.totensor(teacher_pred_bboxes)
                teacher_pred_labels_ = at.totensor(teacher_pred_labels)
                teacher_pred_scores_ = at.totensor(teacher_pred_scores)
                teacher_pred_features_ = at.totensor(teacher_pred_features_)
                # teacher_pred_stage_features1_ = at.totensor(teacher_pred_stage_features1_)
                # teacher_pred_stage_features2_ = at.totensor(teacher_pred_stage_features2_)
                # teacher_pred_stage_features3_ = at.totensor(teacher_pred_stage_features3_)
                teacher_pred_stage_features4_ = at.totensor(teacher_pred_stage_features4_)
                # 使用GPU
                teacher_pred_bboxes_ = teacher_pred_bboxes_.cuda()
                teacher_pred_labels_ = teacher_pred_labels_.cuda()
                teacher_pred_scores_ = teacher_pred_scores_.cuda()
                teacher_pred_features_ = teacher_pred_features_.cuda()
                # teacher_pred_stage_features1_ = teacher_pred_stage_features1_.cuda()
                # teacher_pred_stage_features2_ = teacher_pred_stage_features2_.cuda()
                # teacher_pred_stage_features3_ = teacher_pred_stage_features3_.cuda()
                teacher_pred_stage_features4_ = teacher_pred_stage_features4_.cuda()
                # 如果dataset.py 中的Transform 设置了图像翻转,就要使用这个判读软标签是否一起翻转
                if(teacher_pred_bboxes_[0][1] != bbox[0][0][1]):
                    _, o_C, o_H, o_W = img.shape
                    teacher_pred_bboxes_ = flip_bbox(
                        teacher_pred_bboxes_, (o_H, o_W), x_flip=True)

                losses = trainer.train_step(img, bbox, label, scale, epoch,
                                            teacher_pred_bboxes_, teacher_pred_labels_, teacher_pred_features_, teacher_pred_scores_,
                                            teacher_pred_stage_features4_)
            else:
                trainer.train_step(img, bbox, label, scale, epoch)

        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{},ap:{}, map:{},loss:{}'.format(str(lr_),
                                                        str(eval_result['ap']),
                                                        str(eval_result['map']),
                                                        str(trainer.get_meter_data()))
        print(log_info)

        # 保存最好结果并记住路径
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map, epoch=epoch)

        # 每10轮加载前面最好权重，并且减少学习率
        if epoch % 10 == 0 and epoch != 0:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay


if __name__ == '__main__':
    #train()
    import fire

    fire.Fire()
