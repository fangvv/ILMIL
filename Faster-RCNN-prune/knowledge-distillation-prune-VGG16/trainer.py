from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch as t
from utils import array_tool as at
from utils.vis_tool import Visualizer

from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter
from utils.cfg_mask import get_featureMap, clear_featureMap, get_mask_FM, FM_3D_to_2D, st_add_get_featureMap4, st_add_get_featureMap5, add_clear_featureMap

if opt.use_hint and opt.is_distillation:
    LossTuple = namedtuple('LossTuple',
                           ['rpn_loc_loss',
                            'rpn_cls_loss',
                            'roi_loc_loss',
                            'roi_cls_loss',
                            'hint_loss',
                            'scores_loss',
                            'total_loss'
                            ])
else:
    LossTuple = namedtuple('LossTuple',
                           ['rpn_loc_loss',
                            'rpn_cls_loss',
                            'roi_loc_loss',
                            'roi_cls_loss',
                            'total_loss'
                            ])


class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        # visdom wrapper
        self.vis = Visualizer(env=opt.env)

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxes, labels, scale, epoch,
                teacher_pred_bboxes_, teacher_pred_labels_,
                teacher_stage4_, teacher_stage5_,
                teacher_pred_scores_,faster_rcnn_teacher):

        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        # **************************学生模型的输出信息****************************
        features = self.faster_rcnn.extractor(imgs)  # 对应的教师是teacher_pred_features_

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        if opt.only_use_cls_distillation:
            bbox = bboxes[0]
            label = labels[0]
        else:
            bbox = teacher_pred_bboxes_
            label = teacher_pred_labels_

        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label, gt_roi_score = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            at.tonumpy(teacher_pred_scores_),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)


        # ****************************教师模型的输出信息*****************************

        # ********************************** RPN losses ************************************** #
        # 分数损失
        roi_score_test = roi_score.data
        prob = F.softmax(at.totensor(roi_score_test), dim=1)
        pred_scores, _ = t.max(prob, 1)
        gt_roi_score = at.totensor(gt_roi_score)
        gt_roi_score = gt_roi_score.cuda()

        scores_loss = F.l1_loss(pred_scores, gt_roi_score)

        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        if len(teacher_pred_bboxes_) != 0 and opt.is_distillation == True:
            # 教师的RPN响应
            teacher_rpn_loc, teacher_rpn_label = self.anchor_target_creator(
                at.tonumpy(teacher_pred_bboxes_),
                anchor,
                img_size)
            teacher_rpn_label = at.totensor(teacher_rpn_label).long()
            teacher_rpn_loc = at.totensor(teacher_rpn_loc)
            # 教师的RPN分类损失
            teacher_rpn_cls_loss = F.cross_entropy(
                rpn_score, teacher_rpn_label.cuda(), ignore_index=-1)
            # RPN硬分类损失 + RPN的软分类损失（蒸馏了）---********
            rpn_cls_loss = teacher_rpn_cls_loss

        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ********************************** ROI losses (fast rcnn loss) ************************************** #
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        # *************************** RPN losses and ROI losses(distillation) ********************************
        if opt.use_hint and opt.is_distillation:
            st_stage4 = st_add_get_featureMap4()
            st_stage5 = st_add_get_featureMap5()
            hint_loss = 0
            hint_loss += l2_loss_stage(st_stage4, teacher_stage4_)
            hint_loss += l2_loss_stage(st_stage5, teacher_stage5_)

            clear_featureMap()
            add_clear_featureMap()
            # print("==================")
            # print(rpn_loc_loss)
            # print(rpn_cls_loss)
            # print(roi_loc_loss)
            # print(roi_cls_loss)
            # print(hint_loss)
            # print("=====================")
            losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss,
                      roi_cls_loss,
                      hint_loss,
                      scores_loss
                      ]
        else:
            losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss,
                      roi_cls_loss
                      ]
        losses = losses + [sum(losses)]
        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale, epoch, teacher_pred_bboxes_=None, teacher_pred_labels_=None,
                   teacher_stage4_ = None,teacher_stage5_=None,
                   teacher_pred_scores_={}, faster_rcnn_teacher=None):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale, epoch,
                              teacher_pred_bboxes_, teacher_pred_labels_,
                              teacher_stage4_,teacher_stage5_,
                              teacher_pred_scores_, faster_rcnn_teacher)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, epoch=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            save_path += '_%s' % epoch
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

def l2_loss(gt, pred):
    B, H, W = gt.size()
    # loss = t.sum(t.abs(gt - pred))
    loss = t.sum((gt - pred) * (gt - pred)) / (B * H * W)
    return loss
    # B, C, H, W = gt.size()
    # # loss = t.sum(t.abs(gt - pred))
    # loss = t.sum((gt - pred) * (gt - pred)) / (B * C * H * W)
    # return loss

def l2_loss_stage(gt, pred):
    B, C, H, W = gt.size()
    # loss = t.sum(t.abs(gt - pred))
    gt_mean=t.mean(gt)
    pred_mean=t.mean(pred)
    loss = t.sum((gt_mean - pred_mean) * (gt_mean - pred_mean)) / (B * C * H * W)
    return loss

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss

def teacher_bounded_regression_loss(y, teacher_pred, student_pred):
    zero_loss = t.zeros([1])
    y_teacher = teacher_bounded_regression_l2_loss(y, teacher_pred)
    y_student = teacher_bounded_regression_l2_loss(y, student_pred)
    if y_student > y_teacher:
        return y_student
    else:
        return zero_loss.cuda()

def teacher_bounded_regression_l2_loss(gt, pred):
    H, W = gt.size()
    # loss = t.sum(t.abs(gt - pred))
    loss = t.sum((gt - pred) * (gt - pred)) / ( H * W)
    return loss
    # B, C, H, W = gt.size()
    # # loss = t.sum(t.abs(gt - pred))
    # loss = t.sum((gt - pred) * (gt - pred)) / (B * C * H * W)
    # return loss