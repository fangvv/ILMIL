import os
import xml.etree.ElementTree as ET

import numpy as np

from .util import read_image
from utils.config import opt


class VOCBboxDataset:
    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,
                 ):
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = opt.VOC_BBOX_LABEL_NAMES
        print("=======================VOCBboxDataset==========================")
        print(self.label_names)

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            name = obj.find('name').text.lower().strip()
            if name not in opt.VOC_BBOX_LABEL_NAMES:
                continue
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])

            label.append(opt.VOC_BBOX_LABEL_NAMES_test.index(name))
        if len(bbox) > 0:
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
            difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)
        return img, bbox, label, difficult, id_

    __getitem__ = get_example


class VOCBboxDataset_test:
    # 加载数据的类函数
    def __init__(self, data_dir, split=opt.datatxt,
                 use_difficult=False, return_difficult=False,
                 ):

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = opt.VOC_BBOX_LABEL_NAMES_test
        print("=======================VOCBboxDataset_test==========================")
        print(self.label_names)

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        # 获得图像数据和标注数据

        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        flag = 0
        # 一张图像有多个bbox标签，循环读取bbox标签
        for obj in anno.findall('object'):
            name = obj.find('name').text.lower().strip()
            if name not in opt.VOC_BBOX_LABEL_NAMES_test:
                continue

            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                    int(bndbox_anno.find(tag).text) - 1
                    for tag in ('ymin', 'xmin', 'ymax', 'xmax')])

            label.append(opt.VOC_BBOX_LABEL_NAMES_test.index(name))

        # np.stack在axis=i上增加一维。这边似乎只是转为np型数组
        # 由于这边要做类别增量，可能出现图像没有bbox，所有加一个判断
        # 使得没有bbox的图像不会报错 而是在后面的操作中会跳过
        if len(bbox) > 0:
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)

            difficult = np.array(difficult, dtype=np.bool).astype(
                np.uint8)  # PyTorch don't support np.bool

        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        return img, bbox, label, difficult, id_

    __getitem__ = get_example