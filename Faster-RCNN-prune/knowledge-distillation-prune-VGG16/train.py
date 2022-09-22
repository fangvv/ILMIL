from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize, Transform
from model import FasterRCNNVGG16
from model import FasterRCNNVGG16_PRUNE
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.eval_tool import eval_detection_voc
from data.util import read_image
import numpy as np
import torch
from utils.utils import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import resource
from utils.cfg_mask import farward_hook1, farward_hook2, farward_hook3, farward_hook4, farward_hook5, \
    te_farward_hook1, te_farward_hook2, te_farward_hook3, te_farward_hook4, te_farward_hook5, clear_featureMap,\
    te_add_farward_hook4, te_add_farward_hook5,st_add_farward_hook4,st_add_farward_hook5,add_clear_featureMap

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

def eval_te(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, id_) in tqdm(enumerate(dataloader)):
        if len(gt_bboxes_) == 0:
            continue
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_, _ = faster_rcnn.predict(imgs, [sizes])
        clear_featureMap()
        add_clear_featureMap()
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

def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, id_) in tqdm(enumerate(dataloader)):
        if len(gt_bboxes_) == 0:
            continue
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        clear_featureMap()
        add_clear_featureMap()
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

def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       #pin_memory=True
                                   )
    # *********************************教师模型***************************************

    opt.cfg = opt.tea_cfg
    opt.student=False
    tsf = Transform(opt.min_size, opt.max_size)

    faster_rcnn_teacher = FasterRCNNVGG16()
    faster_rcnn_teacher.extractor[2].register_forward_hook(te_farward_hook1)
    faster_rcnn_teacher.extractor[7].register_forward_hook(te_farward_hook2)
    faster_rcnn_teacher.extractor[14].register_forward_hook(te_farward_hook3)
    faster_rcnn_teacher.extractor[21].register_forward_hook(te_farward_hook4)
    faster_rcnn_teacher.extractor[28].register_forward_hook(te_farward_hook5)

    faster_rcnn_teacher.extractor[23].register_forward_hook(te_add_farward_hook4)
    faster_rcnn_teacher.extractor[29].register_forward_hook(te_add_farward_hook5)
    print("================教师模型====================")
    print(faster_rcnn_teacher)
    print('model construct completed')
    trainer_teacher = FasterRCNNTrainer(faster_rcnn_teacher).cuda()
    trainer_teacher.eval()
    if opt.load_path:
        trainer_teacher.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    
    
    # 提取蒸馏知识所需要的软标签
    # '''
    if opt.is_distillation == True:
        opt.predict_socre = 0.3
        a = 0
        for ii, (imgs, sizes, gt_bboxes_, gt_labels_, scale, id_) in tqdm(enumerate(dataloader)):
            if len(gt_bboxes_) == 0:
                continue
            img, bbox, label = imgs.cuda().float(), gt_bboxes_.cuda(), gt_labels_.cuda()
            sizes = [sizes[0][0].item(), sizes[1][0].item()]
            # 进入的是faster_rcnn_te的predict函数
            pred_bboxes_, pred_labels_, pred_scores_, teacher_stage4, teacher_stage5= trainer_teacher.faster_rcnn.predict(imgs, [sizes], bbox, label)

            img_file = os.path.join(opt.voc_data_dir, 'JPEGImages', id_[0] + '.jpg')
            ori_img = read_image(img_file, color=True)
            img, pred_bboxes_, pred_labels_, scale_ = tsf((ori_img, pred_bboxes_[0], pred_labels_[0]))

            # 去除软标签和真值标签重叠过多的部分，去除错误的软标签
            pred_bboxes_, pred_labels_, pred_scores_ = py_cpu_nms(
                        gt_bboxes_[0], gt_labels_[0], pred_bboxes_, pred_labels_, pred_scores_[0])
            # 存储软标签，这样存储不会使得GPU占用过多
            np.save('label/' + str(id_[0]) + '.npy', pred_labels_.cpu())
            np.save('bbox/' + str(id_[0]) + '.npy', pred_bboxes_.cpu())
            np.save('score/' + str(id_[0]) + '.npy', pred_scores_.cpu())
            np.save('stage4/' + str(id_[0]) + '.npy', teacher_stage4.cpu())
            np.save('stage5/' + str(id_[0]) + '.npy', teacher_stage5.cpu())

        opt.predict_socre = 0.05
        print(a)
    torch.cuda.empty_cache()
    # '''

    # **************************************学生模型***************************************
    opt.student=True
    # opt.cfg = [52, 52, 'M', 104, 104, 'M', 208, 208, 208, 'M', 416, 416, 416, 'M', 416, 416, 512]
    opt.cfg = [60, 60, 'M', 120, 120, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    faster_rcnn_student = FasterRCNNVGG16_PRUNE()
    faster_rcnn_student.extractor[2].register_forward_hook(farward_hook1)
    faster_rcnn_student.extractor[7].register_forward_hook(farward_hook2)
    faster_rcnn_student.extractor[14].register_forward_hook(farward_hook3)
    faster_rcnn_student.extractor[21].register_forward_hook(farward_hook4)
    faster_rcnn_student.extractor[28].register_forward_hook(farward_hook5)

    faster_rcnn_student.extractor[23].register_forward_hook(st_add_farward_hook4)
    faster_rcnn_student.extractor[29].register_forward_hook(st_add_farward_hook5)
    print('model construct completed')
    trainer_student = FasterRCNNTrainer(faster_rcnn_student).cuda()
    trainer_student.train()
    print("==================学生模型=======================")
    print(faster_rcnn_student)
    if opt.caffe_pretrain:
        trainer_student.load(opt.caffe_pretrain_path)
        print('load student pretrain model from %s' % opt.caffe_pretrain_path)

    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        print('epoch=%d' % epoch)
        trainer_student.reset_meters()
        for ii, (img, sizes, bbox_, label_, scale, id_) in tqdm(enumerate(dataloader)):
            if len(bbox_) == 0:
                continue
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            if opt.is_distillation == True:

                file_exist = os.path.exists('label/' + str(id_[0]) + '.npy')
                # if file_exist == False:
                #     continue
                # 读取软标签
                teacher_pred_labels = np.load('label/' + str(id_[0]) + '.npy')
                teacher_pred_bboxes = np.load('bbox/' + str(id_[0]) + '.npy')
                teacher_pred_scores = np.load('score/' + str(id_[0]) + '.npy')
                teacher_stage4_ = np.load('stage4/' + str(id_[0]) + '.npy')
                teacher_stage5_ = np.load('stage5/' + str(id_[0]) + '.npy')
                # 格式转换
                teacher_pred_bboxes = teacher_pred_bboxes.astype(np.float32)
                teacher_pred_labels = teacher_pred_labels.astype(np.int32)
                teacher_pred_scores = teacher_pred_scores.astype(np.float32)
                # 转成pytorch格式
                teacher_pred_bboxes_ = at.totensor(teacher_pred_bboxes)
                teacher_pred_labels_ = at.totensor(teacher_pred_labels)
                teacher_pred_scores_ = at.totensor(teacher_pred_scores)
                teacher_stage4_ = at.totensor(teacher_stage4_)
                teacher_stage5_ = at.totensor(teacher_stage5_)
                # 使用GPU
                teacher_pred_bboxes_ = teacher_pred_bboxes_.cuda()
                teacher_pred_labels_ = teacher_pred_labels_.cuda()
                teacher_pred_scores_ = teacher_pred_scores_.cuda()
                teacher_stage4_ = teacher_stage4_.cuda()
                teacher_stage5_ = teacher_stage5_.cuda()
                # 如果dataset.py 中的Transform 设置了图像翻转,就要使用这个判读软标签是否一起翻转
                # if teacher_pred_bboxes_.size()[0] is 0:
                #     continue
                if (teacher_pred_bboxes_[0][1] != bbox[0][0][1]):
                    _, o_C, o_H, o_W = img.shape
                    teacher_pred_bboxes_ = flip_bbox(teacher_pred_bboxes_, (o_H, o_W), x_flip=True)
                # if teacher_pred_labels_.size()[0] != 128:
                #     continue

                losses = trainer_student.train_step(img, bbox, label, scale, epoch,
                                            teacher_pred_bboxes_, teacher_pred_labels_,
                                            teacher_stage4_, teacher_stage5_, teacher_pred_scores_,faster_rcnn_teacher)
            else:
                trainer_student.train_step(img, bbox, label, scale, epoch)
        eval_result = eval(test_dataloader, faster_rcnn_student, test_num=opt.test_num)
        lr_ = trainer_student.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{}, ap:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(eval_result['ap']),
                                                  str(trainer_student.get_meter_data()))
        print(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer_student.save(best_map=best_map, epoch=epoch)
        if epoch % 5 == 0 and epoch != 0:
            trainer_student.load(best_path)
            trainer_student.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay



if __name__ == '__main__':
    import fire

    fire.Fire()
