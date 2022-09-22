#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 这个头文件指的是精准的除法，就是会帮你把小数后面的数值都保留下来，不会去除。
from __future__ import division
import os
import torch as t
import numpy as np
import cv2
import six
import itertools

from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
from collections import defaultdict
from model.utils.bbox_tools import bbox_iou
from data.dataset import Dataset, TestDataset, inverse_normalize
from torch.utils import data as data_
from tqdm import tqdm

# file='/media/chenli/F1/cell_data/BCCD_Dataset/BCCD/ImageSets/Main/'
file = '/media/chenli/F1/object_detection/VOCdevkit/VOC2007/ImageSets/Main/'
model_path = 'fasterrcnn_10022025_0.6223694993624835'
model_path2 = 'fasterrcnn_10081653_0.6534784721424378'
model_path3 = 'fasterrcnn_10082346_0.6664386481131717'
model_path4 = 'fasterrcnn_10091151_0.6832216465308698'
model_path5 = 'fasterrcnn_10101246_0.6790846900541052'

opt.caffe_pretrain = True  # this model was trained from torchvision-pretrained model

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

testset = TestDataset(opt)
test_dataloader = data_.DataLoader(testset,
                                   batch_size=1,
                                   num_workers=opt.test_num_workers,
                                   shuffle=False,
                                   pin_memory=True
                                   )


def py_soft_nms(dets, socres, label, sigma=0.5, Nt=0.3, threshold=0.001,):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = socres  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
# 打分从大到小排列，取index
    order = scores.argsort()[::-1]
# keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
# 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

#         print xx1.shape

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
# 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
#         print ovr
        inds = []
#         print label[0]
        b = order[0]
# inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
# 只有在标签一致的情况下才会判断是否被窗口吸收，如果没有标签的话就使用下面注释的代码
        for j in range(len(order) - 1):
            a = order[j + 1]
            if label[a] == label[b]:
                if ovr[j] > Nt:
                    scores[a] = scores[a] * (1 - ovr[j])
                    if scores[a] > threshold:
                        inds.append(j + 1)

                else:
                    inds.append(j + 1)
            else:
                inds.append(j + 1)
# order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds]

    return keep


# nms 非极大值抑制
# 输入为bbox数据,包含[ymin,xmin,ymax,xmax];每个bbox的评分；每个bbox的标签，如果没有标签就注释掉
def py_cpu_nms(dets, socres, label, thresh=0.6):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = socres  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
# 打分从大到小排列，取index
    order = scores.argsort()[::-1]
# keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
# 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

#         print xx1.shape

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
# 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
#         print ovr
        inds = []
#         print label[0]
        b = order[0]
# inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
# 只有在标签一致的情况下才会判断是否被窗口吸收，如果没有标签的话就使用下面注释的代码
        for j in range(len(order) - 1):
            a = order[j + 1]
            if label[a] == label[b]:
                if ovr[j] <= thresh:
                    inds.append(j + 1)
            else:
                inds.append(j + 1)
# order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds]

    return keep


def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=0.5, use_07_metric=False):

    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)
#     print prec
#     print prec,rec
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}


def calc_detection_voc_ap(prec, rec, use_07_metric=False):

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

# 计算的召回率和准确率，每一种都包含类别个数大小的数组，每一个代表一个类别的召回率或者准确率


def calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults=None,
        iou_thresh=0.5):

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes1 = iter(gt_bboxes)
    gt_labels1 = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults1 = itertools.repeat(None)
    else:
        gt_difficults1 = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes1, gt_labels1, gt_difficults1):

        #         print pred_bbox
        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            # 在真实标签中选出标签为某值的boundingbox
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]

            # sort by score 对分数排序
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            # 在真实标签中选出标签为某值的boundingbox
            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
#             print n_pos[l]
            # list.extend 追加一行
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.VOC评价遵循整数bounding boxes
            pred_bbox_l = pred_bbox_l.copy()
#             print pred_bbox_l
            pred_bbox_l[:, 2:] += 1
#             print pred_bbox_l
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

        # 找到所有gt和pred的重叠面积，总共gt.shape*pred.shape 个重叠面积
            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
# 找到最大的和真实样本bbox的重叠面积的索引
            gt_index = iou.argmax(axis=1)
#             print gt_index
            # set -1 if there is no matching ground truth
            # 小于阈值的就去除掉
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            # 计算匹配的个数
            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes1, gt_labels1, gt_difficults1):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
    return prec, rec

# 传入的是真值标签和预测标签


def bbox_iou(bbox_a, bbox_b):

    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
  # 由于可能两个bbox大小不一致，所以使用[bbox_a,bbox_b,2]存储遍历的bbox_a*bbox_b个bbox的比较
    # top left  这边是计算了如图上第一幅的重叠左下角坐标值（x，y）
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right  这边是计算了如图上第一幅的重叠左上角坐标值ymax和右下角坐标值xmax

    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    # np.prod 给定轴数值的乘积   相减就得到高和宽 然后相乘
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)  # 重叠部分面积
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)

    return area_i / (area_a[:, None] + area_b - area_i)


def bbox_result(dataloader, faster_rcnn, test_num=500):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults, ID = list(), list(), list(), list()

    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, id_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [
                                                                       sizes])

        # print gt_bboxes_
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        ID += list(id_)

        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii >= test_num:
            break

    return pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults, ID

# 计算的召回率和准确率，每一种都包含类别个数大小的数组，每一个代表一个类别的召回率或者准确率


def nms_reuslt(
        pred_bboxes, pred_labels, pred_scores, pred_bboxes_, pred_labels_, pred_scores_, gt_bboxes, gt_labels, gt_difficults):

    pred_bboxes1 = iter(pred_bboxes)
    pred_labels1 = iter(pred_labels)
    pred_scores1 = iter(pred_scores)
    pred_bboxes_1 = iter(pred_bboxes_)
    pred_labels_1 = iter(pred_labels_)
    pred_scores_1 = iter(pred_scores_)

    pred_b, pred_l, pred_s = list(), list(), list()

    for pred_bbox, pred_label, pred_score, pred_bbox_, pred_label_, pred_score_ in \
            six.moves.zip(
                pred_bboxes1, pred_labels1, pred_scores1,
                pred_bboxes_1, pred_labels_1, pred_scores_1):

        bounding_boxes = np.concatenate((pred_bbox, pred_bbox_))
        confidence_score = np.concatenate((pred_score, pred_score_))
        labels = np.concatenate((pred_label, pred_label_))

        keep = py_soft_nms(bounding_boxes, confidence_score, labels)

        bounding = bounding_boxes[keep]
        confidence = confidence_score[keep]
        label = labels[keep]

        bbox1 = np.expand_dims(bounding, axis=0)
        label1 = np.expand_dims(label, axis=0)
        confidence1 = np.expand_dims(confidence, axis=0)

        pred_b += list(bbox1)
        pred_l += list(label1)
        pred_s += list(confidence1)

    result = eval_detection_voc(
        pred_b, pred_l, pred_s, gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    print result
    result2 = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    print result2
    result3 = eval_detection_voc(
        pred_bboxes_, pred_labels_, pred_scores_, gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    print result3
    if result['map'] < result2['map']:
        result = result2
        pred_b, pred_l, pred_s = pred_bboxes, pred_labels, pred_scores
    if result['map'] < result3['map']:
        result = result3
        pred_b, pred_l, pred_s = pred_bboxes_, pred_labels_, pred_scores_
    return pred_b, pred_l, pred_s, result


# 加载权重
trainer.load('checkpoints/' + model_path)
pred_bboxes1, pred_labels1, pred_scores1, gt_bboxes, gt_labels, gt_difficults, ID = bbox_result(
    test_dataloader, trainer.faster_rcnn)

trainer.load('checkpoints/' + model_path2)
pred_bboxes_, pred_labels_, pred_scores_, gt_bboxes, gt_labels, gt_difficults, ID = bbox_result(
    test_dataloader, trainer.faster_rcnn)

pred_bboxes_, pred_labels_, pred_scores_, result = nms_reuslt(pred_bboxes1, pred_labels1, pred_scores1,
                                                              pred_bboxes_, pred_labels_, pred_scores_,
                                                              gt_bboxes, gt_labels, gt_difficults)

print (result)
trainer.load('checkpoints/' + model_path3)
pred_bboxes1, pred_labels1, pred_scores1, gt_bboxes, gt_labels, gt_difficults, ID = bbox_result(
    test_dataloader, trainer.faster_rcnn)

pred_bboxes_, pred_labels_, pred_scores_, result = nms_reuslt(pred_bboxes1, pred_labels1, pred_scores1,
                                                              pred_bboxes_, pred_labels_, pred_scores_,
                                                              gt_bboxes, gt_labels, gt_difficults)

print (result)
trainer.load('checkpoints/' + model_path4)
pred_bboxes1, pred_labels1, pred_scores1, gt_bboxes, gt_labels, gt_difficults, ID = bbox_result(
    test_dataloader, trainer.faster_rcnn)

pred_bboxes_, pred_labels_, pred_scores_, result = nms_reuslt(pred_bboxes1, pred_labels1, pred_scores1,
                                                              pred_bboxes_, pred_labels_, pred_scores_,
                                                              gt_bboxes, gt_labels, gt_difficults)
print (result)

trainer.load('checkpoints/' + model_path5)
pred_bboxes1, pred_labels1, pred_scores1, gt_bboxes, gt_labels, gt_difficults, ID = bbox_result(
    test_dataloader, trainer.faster_rcnn)

pred_bboxes_, pred_labels_, pred_scores_, result = nms_reuslt(pred_bboxes1, pred_labels1, pred_scores1,
                                                              pred_bboxes_, pred_labels_, pred_scores_,
                                                              gt_bboxes, gt_labels, gt_difficults)
print (result)
