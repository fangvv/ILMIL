#从原始的权重文件------转换为----各个层的Tensor
from torch import IntTensor
from torch.autograd import Variable
import torch

#提取模型特定卷积层的权重，并通过纬度转换函数将其转换成特定纬度，最终返回一个包含所有层处理后的权重（list）类型
def weightToTensor(weight):
    list=[]
    with torch.no_grad():
        for name, param in weight.items():
            if "module.backbone.body.layer1.0.conv1.weight" in name:
                list = [fourTotwo(param)]
            elif "module.backbone.body.layer1.0.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer1.0.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer1.0.downsample.0.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer1.1.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer1.1.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer1.1.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer1.2.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer1.2.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer1.2.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer2.0.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer2.0.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer2.0.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.ackbone.body.layer2.0.downsample.0.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer2.1.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer2.1.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer2.1.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer2.2.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer2.2.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer2.3.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer2.3.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer2.3.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.0.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.0.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.0.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.0.downsample.0.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.1.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.1.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.1.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.2.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.2.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.2.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.3.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.3.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.3.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.4.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.4.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.4.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.5.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.5.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer3.5.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer4.0.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer4.0.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer4.0.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer4.0.downsample.0.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer4.1.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer4.1.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer4.1.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer4.2.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer4.2.conv2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.layer4.2.conv3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.body.stem.conv1.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.fpn.fpn_inner2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.fpn.fpn_inner3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.fpn.fpn_inner4.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.fpn.fpn_layer2.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.fpn.fpn_layer3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.fpn.fpn_layer4.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.fpn.top_blocks.p6.weight" in name:
                list.append(fourTotwo(param))
            elif "module.backbone.fpn.top_blocks.p7.weight" in name:
                list.append(fourTotwo(param))
            elif "module.rpn.head.bbox_pred.weight" in name:
                list.append(fourTotwo(param))
            elif "module.rpn.head.bbox_tower.0.weight" in name:
                list.append(fourTotwo(param))
            elif "module.rpn.head.bbox_tower.3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.rpn.head.bbox_tower.6.weight" in name:
                list.append(fourTotwo(param))
            elif "module.rpn.head.bbox_tower.9.weight" in name:
                list.append(fourTotwo(param))
            elif "module.rpn.head.centerness.weight" in name:
                list.append(fourTotwo(param))
            # elif "module.rpn.head.cls_logits.weight" in name:
            #     list.append(fourTotwo(param[0:40,:,:]))
            #     #list = list + fourTotwo(param)
            elif "module.rpn.head.cls_tower.0.weight" in name:
                list.append(fourTotwo(param))
            elif "module.rpn.head.cls_tower.3.weight" in name:
                list.append(fourTotwo(param))
            elif "module.rpn.head.cls_tower.6.weight" in name:
                list.append(fourTotwo(param))
            elif "module.rpn.head.cls_tower.9.weight" in name:
                list.append(fourTotwo(param))
    return list

#进行维度转换操作
def fourTotwo(param):
    _weight = param
    _tensor = _weight.view([_weight.size()[0], -1])  # 将（64，64，3，3）-----转为（64，576）
    _list = _tensor.cpu().numpy().tolist()
    return _list