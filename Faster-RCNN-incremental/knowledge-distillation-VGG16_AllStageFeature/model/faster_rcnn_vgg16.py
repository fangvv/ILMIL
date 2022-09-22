#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
import torch as t
from torch import nn
#from torchvision.models import vgg16
from model.VGG import VGG16 as vgg16
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain: # 有预训练模型
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
            print('load caffe_pretrain_path')
    else:
        model = vgg16(not opt.load_path)
    middle_features1 = list(model.features1)[:4]
    middle_pooling1 = list(model.features1)[4:5]
    middle_features2 = list(model.features2)[:4]
    middle_pooling2 = list(model.features2)[4:5]
    middle_features3 = list(model.features3)[:6]
    middle_pooling3 = list(model.features3)[6:7]
    middle_features4 = list(model.features4)[:6]
    middle_pooling4 = list(model.features4)[6:7]
    features = list(model.features5)[:6]
    classifier = model.classifier

    classifier = list(classifier)

# 保留分类器的几层  作为最后roi阶段的分类层
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

# 冻结卷积网络前4层
# freeze top4 conv
    for layer in middle_features1[:]:
        for p in layer.parameters():
            p.requires_grad = False
    for layer in middle_features2[:]:
        for p in layer.parameters():
            p.requires_grad = False
    return nn.Sequential(*features), classifier, nn.Sequential(*middle_features1), nn.Sequential(*middle_features2), \
           nn.Sequential(*middle_features3), nn.Sequential(*middle_features4),\
            nn.Sequential(*middle_pooling1), nn.Sequential(*middle_pooling2), nn.Sequential(*middle_pooling3), nn.Sequential(*middle_pooling4)


class FasterRCNNVGG16(FasterRCNN):
    # 分为3部分，首先是提取特征，其次是rpn，最后是roi
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):

        extractor, classifier, middle_extractor1, middle_extractor2, middle_extractor3, middle_extractor4, middle_pooling1, middle_pooling2, middle_pooling3, middle_pooling4 = decom_vgg16()
        # extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor, middle_extractor1, middle_extractor2, middle_extractor3, middle_extractor4,
            middle_pooling1, middle_pooling2, middle_pooling3, middle_pooling4,
            # extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size

        # 这边是1/16
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(
            self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        # 将roi和roi_indices拼接 形成[yx]的结构，y是roi的标签
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)

        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        # 对特征图进行空间金字塔池化
        pool = self.roi(x, indices_and_rois)
        # 平铺
        pool = pool.view(pool.size(0), -1)

        # 分类层，是截取部分的vgg16分类层，获得多通道的特征图
        fc7 = self.classifier(pool)

        # 获得roi4个坐标和分数
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
            mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
