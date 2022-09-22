#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from tqdm import tqdm
from .eval_tool import eval_detection_voc
from utils import array_tool as at
import torch as t


def flip_bbox(bbox, size, x_flip=False):
    # 水平翻转
    H, W = size
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox

def py_cpu_nms(gt_bboxes_, gt_labels_, teacher_pred_bboxes_, teacher_pred_labels_, teacher_pred_scores_, thresh=0.7):
    # 使用nms判断预测bbox是否与真标签iou大小接近,如果接近就去除
    x1 = gt_bboxes_[:, 0]
    y1 = gt_bboxes_[:, 1]
    x2 = gt_bboxes_[:, 2]
    y2 = gt_bboxes_[:, 3]

    x1_pre = teacher_pred_bboxes_[:, 0]
    y1_pre = teacher_pred_bboxes_[:, 1]
    x2_pre = teacher_pred_bboxes_[:, 2]
    y2_pre = teacher_pred_bboxes_[:, 3]

    # scores = socres  # bbox打分

    areas = (x2_pre - x1_pre + 1) * (y2_pre - y1_pre + 1)
    areas2 = (x2 - x1 + 1) * (y2 - y1 + 1)
    # keep为最后保留的边框
    keep = []
    inds = []
    # print pred_scores[0]
    for i in range(len(teacher_pred_bboxes_)):
        if teacher_pred_scores_[i] <= 0.5:
            flag = 1
            continue
        flag = 0
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1_pre[i], x1)
        yy1 = np.maximum(y1_pre[i], y1)
        xx2 = np.minimum(x2_pre[i], x2)
        yy2 = np.minimum(y2_pre[i], y2)

#         print xx1.shape

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas2 - inter)

        for j in range(len(gt_bboxes_)):
            if ovr[j] >= thresh:
                flag = 1
                # print 1
                break
        if flag == 0:
            inds.append(i)

    gt_scores_ = t.ones(gt_labels_.shape)

    teacher_pred_bboxes = teacher_pred_bboxes_[inds]
    teacher_pred_labels = teacher_pred_labels_[inds]
    teacher_pred_scores = teacher_pred_scores_[inds]

    teacher_pred_bboxes = teacher_pred_bboxes.astype(
        np.float32)
    teacher_pred_labels = teacher_pred_labels.astype(np.int32)
    teacher_pred_scores = teacher_pred_scores.astype(
        np.float32)

    teacher_pred_bboxes_ = at.totensor(teacher_pred_bboxes)
    teacher_pred_labels_ = at.totensor(teacher_pred_labels)
    teacher_pred_scores_ = at.totensor(teacher_pred_scores)
    gt_bboxes_ = gt_bboxes_.cuda()
    gt_labels_ = gt_labels_.cuda()
    gt_scores_ = gt_scores_.cuda()
    gt_bboxes_ = t.cat((gt_bboxes_, teacher_pred_bboxes_))
    gt_labels_ = t.cat((gt_labels_, teacher_pred_labels_))
    gt_scores_ = t.cat((gt_scores_, teacher_pred_scores_))
    return gt_bboxes_, gt_labels_, gt_scores_

def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, id_) in tqdm(enumerate(dataloader)):
        if len(gt_bboxes_) == 0:
            continue
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_, _ = faster_rcnn.predict(imgs, [
            sizes])

        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num:
            break

    # 这个评价函数是返回ap 和map值 其中传入的pred_bboxes格式为3维的数组的list格式，
    # 也就是说每个list都是一个3维数组(有batch的考量)
    # 其他的同理

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result
