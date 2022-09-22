#用于计算两个一维向量的余弦相似度
import torch
import numpy
def cal_cosine_similarity(student_list, teacher_list):
    similarity_lists = []
    for i in range(len(student_list)):
        similarity_list = []
        for j in range(len(student_list[i])):#student_list[i](64)(125)
            student_list_= student_list[i]
            teacher_list_ = teacher_list[i]
            student_tensor = torch.from_numpy(numpy.array(student_list_[j]))#numpy将list转成ndarray，torch.from_numpy将ndarray转成tensor
            teacher_tensor = torch.from_numpy(numpy.array(teacher_list_[j]))
            similarity = torch.cosine_similarity(student_tensor, teacher_tensor, dim=0)
            #print (len(similarity))
            similarity_list.append(similarity.cpu().numpy().tolist())
        similarity_lists.append(similarity_list)
        #print ("-------------------------------------"+str(len(similarity_list)))
    return similarity_lists#大小为67
    #print ("-------------------------------------" + str(len(similarity_lists)))