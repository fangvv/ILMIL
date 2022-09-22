'''本程序用于验证hook编程获取卷积层的输出特征图和特征图的梯度;获取中间重要的feature map'''
#-*- utf-8 -*-
import os
import pickle
import numpy as np
import torch

# 读取保存的cfg_mask的信息
def get_cfg_mask(index):
    savepath = os.path.join("pretrained_model", "cfg_mask_15+1.txt")
    fileHandle = open(savepath, 'rb')
    testList = pickle.load(fileHandle)
    fileHandle.close()
    #print(len(testList))  # 13
    return testList[index]


# 装feature map
te_fmap_block1 = dict()
te_fmap_block2 = dict()
te_fmap_block3 = dict()
te_fmap_block4 = dict()
te_fmap_block5 = dict()
fmap_block1 = dict()
fmap_block2 = dict()
fmap_block3 = dict()
fmap_block4 = dict()
fmap_block5 = dict()
te_fmap_list = []
fmap_list = []
te_add4 = []
te_add5 = []
st_add4 = []
st_add5 = []


# 模块，模块的输入，模块的输出
# 函数用于描述对这些参数的操作，一般我们都是为了获取特征图，即只描述对output的操作即可。
# ****************************教师模型的hook函数*******************************
def te_farward_hook1(module, inp, outp):
    te_fmap_block1['output1'] = outp
    te_fmap_list.append(te_fmap_block1)

def te_farward_hook2(module, inp, outp):
    te_fmap_block2['output2'] = outp
    te_fmap_list.append(te_fmap_block2)

def te_farward_hook3(module, inp, outp):
    te_fmap_block3['output3'] = outp
    te_fmap_list.append(te_fmap_block3)

def te_farward_hook4(module, inp, outp):
    te_fmap_block4['output4'] = outp
    te_fmap_list.append(te_fmap_block4)

def te_farward_hook5(module, inp, outp):
    te_fmap_block5['output5'] = outp
    te_fmap_list.append(te_fmap_block5)

# ***********************学生模型的hook函数******************************
def farward_hook1(module, inp, outp):
    fmap_block1['output1'] = outp
    fmap_list.append(fmap_block1)

def farward_hook2(module, inp, outp):
    fmap_block2['output2'] = outp
    fmap_list.append(fmap_block2)

def farward_hook3(module, inp, outp):
    fmap_block3['output3'] = outp
    fmap_list.append(fmap_block3)

def farward_hook4(module, inp, outp):
    fmap_block4['output4'] = outp
    fmap_list.append(fmap_block4)

def farward_hook5(module, inp, outp):
    fmap_block5['output5'] = outp
    fmap_list.append(fmap_block5)


# ********************增量时所需的教师的feature map**************************
def te_add_farward_hook4(module, inp, outp):  # stage4
    te_add4.append(outp)

def te_add_farward_hook5(module, inp, outp):  # stage5
    te_add5.append(outp)

# ********************增量时所需的学生的feature map**************************
def st_add_farward_hook4(module, inp, outp):  # stage4
    st_add4.append(outp)

def st_add_farward_hook5(module, inp, outp):  # stage5
    st_add5.append(outp)

# ********************返回增量时教师模型和学生模型的feature map*************************
def te_add_get_featureMap4():
    return te_add4[0]

def te_add_get_featureMap5():
    return te_add5[0]

def st_add_get_featureMap4():
    return st_add4[0]

def st_add_get_featureMap5():
    return st_add5[0]

def add_clear_featureMap():
    del te_add4[:]
    del te_add5[:]
    del st_add4[:]
    del st_add5[:]

# ********************返回所需要的的所有的feature map**************************
def te_get_featureMap():
    return te_fmap_list

def get_featureMap():
    return fmap_list

def clear_featureMap():
    del te_fmap_list[:]
    del fmap_list[:]



# 输入的是五个stage的所有的feature map，并且存储在一个list中。返回结果是五个stage对应的提取出来的feature
def get_mask_FM(all_feature_list):
    '''
    第1个stage——>1
    第2个stage——>3
    第3个stage——>6
    第4个stage——>9
    第5个stage——>12
    '''
    # print(all_feature_list)
    stage_to_layer = {0: 1, 1: 3, 2: 6, 3: 9, 4: 12}
    new_feature = []
    for stage_index in range(len(all_feature_list)):
        stage_feature = all_feature_list[stage_index]  # 第n个stage的特征图的输出
        feature_mask = get_cfg_mask(stage_to_layer[stage_index])  # 第n个stage对应的层的cfg_mask

        # 通过所有的feature map和mask---->处理得到的重要的feature map
        old_stage_feature = stage_feature['output'+str(stage_index+1)]  # torch.Size([1, 64, 600, 658])
        old_feature_mask = feature_mask  # torch.Size([64])
        idx0 = np.squeeze(np.argwhere(np.asarray(old_feature_mask.cpu().numpy())))  # 上一层的mask，即该层输入的filter的索引值
        new_stage_feature = old_stage_feature[:, idx0.tolist(), :, :].clone()
        new_feature.append(new_stage_feature)
    return new_feature

# 将一阶段的特征图进行压缩，处理为二维的
def FM_3D_to_2D(feature_list_3D):
    '''
    :param 输入的是list，其中保存着五个stage的三维tensor：torch.Size([1, 12, 600, 800])
    :return: 输出是list，其中保存着五个stage的二维tensor：torch.Size([1, 600, 800])
    '''
    feature_list_2D = []
    for index in range(len(feature_list_3D)):
        old_feature = feature_list_3D[index]
        new_feature = torch.mean(old_feature, 1)
        feature_list_2D.append(new_feature)
    return feature_list_2D


