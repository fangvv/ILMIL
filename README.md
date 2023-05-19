# ILMIL

This is the source code for our paper: **基于知识蒸馏的目标检测模型增量深度学习方法**.

> 随着万物互联时代的到来，具备目标检测能力的物联网设备数量呈爆炸式增长。基于此，网络边缘产生了海量的实时数据，具有低时延、低带宽成本和高安全性特点的边缘计算随之成为一种新兴的计算模式。传统的深度学习方法通常假定在模型训练前所有数据已完全具备，然而实际的边缘计算场景中大量的新数据及类别往往随时间逐渐产生和获得。为了在训练数据成批积累和更新的条件下在资源有限的边缘设备上高效率地完成目标检测任务，本文提出了基于多中间层知识蒸馏的增量学习方法（incremental learning method based on knowledge distillation of multiple intermediate layers,ILMIL）。首先，为了能够适当地保留原有数据中的知识，提出了包含多个网络中间层知识的蒸馏指标（multi-layerfeaturemapRPNandRCN knowledge,MFRRK）。ILMIL将教师模型和学生模型的中间层特征的差异加入模型训练，相比于现有的基于知识蒸馏方法的增量学习，采用ILMIL方法训练的学生模型可以从教师模型的中间层学习到更多的旧类信息来缓解遗忘；其次，ILMIL利用MFRRK蒸馏知识完成现有模型的增量训练，避免训练使用多个独立模型带来的资源开销；为进一步降低模型复杂度以高效地在边缘设备上部署推理，可在知识蒸馏前进行剪枝操作来压缩现有模型。通过在不同场景和条件下的实验对比，本文方法可在有效降低模型计算和存储开销的前提下，缓解已有知识的灾难性遗忘现象，并维持可接受的推理精度。

本文发表在工程科学与技术，[链接](https://kns.cnki.net/kcms/detail/detail.aspx?doi=10.15961/j.jsuese.202100925)

## Citation

    @article{方维维2022基于知识蒸馏的目标检测模型增量深度学习方法,
    title={基于知识蒸馏的目标检测模型增量深度学习方法},
    author={方维维 and 陈爱方 and 孟娜 and 程虎威 and 王清立},
    journal={工程科学与技术},
    volume={54},
    number={6},
    pages={59--66},
    year={2022},
    publisher={工程科学与技术}
    }

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.