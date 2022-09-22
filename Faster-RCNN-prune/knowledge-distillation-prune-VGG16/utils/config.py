from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    voc_data_dir = '/home/caf/data/VOCdevkit2007/VOC2007'
    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 8
    test_num_workers = 8
    predict_socre = 0.05
    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.5  # 1e-3 -> 1e-4
    lr = 1e-4


    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 30


    use_adam = False  # Use Adam optimizer
    use_chainer = False  # try match everything as chainer
    use_drop = False  # use dropout in RoIHead
    # debug
    debug_file = './tmp/debugf'
    test_num = 10000
    # model
    is_distillation = True
    only_use_cls_distillation = False
    use_hint = True
    testtxt = 'test'
    datatxt = 'trainval'
    load_path = 'checkpoints/17+1/fasterrcnn_04212213_11_0.5426103274175206'
    student = False
    # cfg = [52, 52, 'M', 104, 104, 'M', 208, 208, 208, 'M', 416, 416, 416, 'M', 416, 416, 512]
    tea_cfg = [60, 60, 'M', 120, 120, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    cfg = [60, 60, 'M', 120, 120, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    caffe_pretrain = True # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/17+1/fasterrcnn_04212213_11_0.5426103274175206'

    VOC_BBOX_LABEL_NAMES = (
        'train')
    index=19
    VOC_BBOX_LABEL_NAMES_test = (
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        # 'tvmonitor'
    )
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
