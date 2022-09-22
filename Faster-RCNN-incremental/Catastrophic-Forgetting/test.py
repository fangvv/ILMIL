from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize, TestDataset_all
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
#CUDA_VISIBLE_DEVICES=1 python3 test.py --test_path="./checkpoints/fasterrcnn_11091854_0.4595028584047549.pth"
import resource
import argparse

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--test_path', dest='test_path',
                      help='directory to load models', default="checkpoints",
                      type=str)
  args = parser.parse_args()
  return args

def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        if len(gt_bboxes_) == 0:
            continue
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
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


def test(**kwargs):
    args = parse_args()
    opt._parse(kwargs)
    opt.test_path = args.test_path
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
                                       pin_memory=True
                                       )
    testset = TestDataset_all(opt)
    test_dataloader_all = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       # pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.test_path:
        trainer.load(opt.test_path)
        print('load pretrained model from %s' % opt.test_path)
    trainer.eval()
    eval_result = eval(test_dataloader_all, faster_rcnn, test_num=opt.test_num)
    lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
    log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                str(eval_result['ap']),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
    print(log_info)


if __name__ == '__main__':
    test()