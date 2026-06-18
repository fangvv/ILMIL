# ILMIL

This is the source code for our paper: **基于知识蒸馏的目标检测模型增量深度学习方法 (Incremental Learning Method based on Knowledge Distillation of Multiple Intermediate Layers)**. A brief introduction of this work is as follows:

> 随着万物互联时代的到来，具备目标检测能力的物联网设备数量呈爆炸式增长。基于此，网络边缘产生了海量的实时数据，具有低时延、低带宽成本和高安全性特点的边缘计算随之成为一种新兴的计算模式。传统的深度学习方法通常假定在模型训练前所有数据已完全具备，然而实际的边缘计算场景中大量的新数据及类别往往随时间逐渐产生和获得。为了在训练数据成批积累和更新的条件下在资源有限的边缘设备上高效率地完成目标检测任务，本文提出了基于多中间层知识蒸馏的增量学习方法（incremental learning method based on knowledge distillation of multiple intermediate layers,ILMIL）。首先，为了能够适当地保留原有数据中的知识，提出了包含多个网络中间层知识的蒸馏指标（multi-layerfeaturemapRPNandRCN knowledge,MFRRK）。ILMIL将教师模型和学生模型的中间层特征的差异加入模型训练，相比于现有的基于知识蒸馏方法的增量学习，采用ILMIL方法训练的学生模型可以从教师模型的中间层学习到更多的旧类信息来缓解遗忘；其次，ILMIL利用MFRRK蒸馏知识完成现有模型的增量训练，避免训练使用多个独立模型带来的资源开销；为进一步降低模型复杂度以高效地在边缘设备上部署推理，可在知识蒸馏前进行剪枝操作来压缩现有模型。通过在不同场景和条件下的实验对比，本文方法可在有效降低模型计算和存储开销的前提下，缓解已有知识的灾难性遗忘现象，并维持可接受的推理精度。

> With the advent of the Internet of Everything era, the number of IoT devices with object detection capabilities has grown exponentially. Consequently, massive amounts of real-time data are generated at the network edge, making edge computing—an emerging computing paradigm characterized by low latency, low bandwidth costs, and high security—increasingly relevant. Traditional deep learning methods typically assume that all data is fully available before model training. However, in real-world edge computing scenarios, large amounts of new data and categories are often generated and acquired gradually over time. To efficiently perform object detection tasks on resource-constrained edge devices under conditions where training data is accumulated and updated in batches, this paper proposes an incremental learning method based on knowledge distillation of multiple intermediate layers (ILMIL). First, to appropriately retain knowledge from existing data, a distillation metric incorporating knowledge from multiple intermediate network layers (multi-layer feature map RPN and RCN knowledge, MFRRK) is introduced. ILMIL incorporates the differences in intermediate layer features between the teacher and student models into the training process. Compared to existing incremental learning methods based on knowledge distillation, the student model trained with ILMIL can learn more information about old classes from the teacher model's intermediate layers, mitigating catastrophic forgetting. Second, ILMIL leverages MFRRK distillation to perform incremental training of the existing model, avoiding the resource overhead of training multiple independent models. To further reduce model complexity for efficient inference deployment on edge devices, pruning operations can be applied to compress the existing model before knowledge distillation. Through experimental comparisons across different scenarios and conditions, the proposed method effectively reduces computational and storage overhead while mitigating catastrophic forgetting of existing knowledge and maintaining acceptable inference accuracy.

本文发表在工程科学与技术，[链接](https://kns.cnki.net/kcms/detail/detail.aspx?doi=10.15961/j.jsuese.202100925)，被评为该刊2024年高影响力论文([证书](https://fangvv.github.io/Homepage/jsuese.pdf))。

## Required software

- PyTorch >= 1.0
- torchvision
- NumPy
- scikit-image
- tqdm
- fire
- matplotlib
- visdom
- Cython
- torchnet

## Project Structure

```
ILMIL/
├── Faster-RCNN-incremental/
│   ├── Catastrophic-Forgetting/                    # Baseline: fine-tuning without KD (catastrophic forgetting)
│   ├── knowledge-distillation-VGG16/               # ILMIL: MFRRK distillation (RPN + RCN features)
│   └── knowledge-distillation-VGG16_AllStageFeature/  # Extension: KD with all intermediate stage features
├── Faster-RCNN-prune/
│   ├── simple-faster-rcnn-prune-VGG16/             # L1-norm based channel pruning on VGG16 backbone
│   └── knowledge-distillation-prune-VGG16/         # Pruning + KD combined approach
└── README.md
```

Each experiment directory follows a consistent internal structure:
```
├── data/                   # Dataset loading (VOC format) and preprocessing
├── model/                  # Core model definitions
│   ├── faster_rcnn.py              # Base Faster R-CNN framework
│   ├── faster_rcnn_vgg16.py        # VGG16-based Faster R-CNN
│   ├── faster_rcnn_vgg16_prune.py  # Pruned VGG16 variant (prune only)
│   ├── region_proposal_network.py  # RPN module
│   ├── roi_module.py               # RoI pooling module
│   ├── vgg.py / VGG.py             # VGG16 backbone
│   └── utils/                      # BBox tools, NMS, anchor/proposal creators
├── utils/                  # Configuration, evaluation, visualization utilities
├── misc/                   # Pretrained model conversion, demo scripts
├── trainer.py              # Training loop with loss computation
├── train.py                # Training entry script
├── test.py                 # Evaluation entry script
└── requirements.txt
```

## Core Modules

### Baseline: Catastrophic Forgetting (`Catastrophic-Forgetting/`)

Directly fine-tunes a pre-trained Faster R-CNN (VGG16 backbone) on new-class data without any knowledge distillation. Serves as the lower-bound baseline demonstrating catastrophic forgetting of old classes.

**Key behavior:**
- The teacher model (trained on old classes) is discarded; only new-class data is used for fine-tuning.
- Old-class AP drops significantly after fine-tuning, validating the need for an incremental learning approach.

### ILMIL: MFRRK Distillation (`knowledge-distillation-VGG16/`)

Implements the core **ILMIL** method with **MFRRK** (Multi-layer Feature map RPN and RCN Knowledge) distillation loss. The teacher model (pre-trained on old classes) guides the student model during incremental training on new classes, using intermediate feature-level distillation to mitigate catastrophic forgetting.

**Key components:**
- `trainer.py` — `FasterRCNNTrainer` computes the full training loss including:
  - `rpn_loc_loss` / `rpn_cls_loss`: Standard RPN regression and classification losses.
  - `roi_loc_loss` / `roi_cls_loss`: Standard RoI head regression and classification losses.
  - `hint_loss`: L2 distillation loss between teacher and student intermediate feature maps (RPN/RCN features), controlled by `opt.use_hint`.
  - `scores_loss`: Classification score-level distillation loss from teacher to student.
- `utils/config.py` — Configuration hub. Key options include:
  - `is_distillation`: Enable knowledge distillation mode.
  - `use_hint`: Enable intermediate feature distillation (MFRRK).
  - `only_use_cls_distillation`: Use only classification-level distillation (ablation).
  - `load_path`: Path to the pre-trained teacher model checkpoint.
- `model/faster_rcnn_vgg16.py` — VGG16-based Faster R-CNN with hooks to extract intermediate features from RPN and RoI head for distillation.
- `nms_Ensemble.py` — Soft-NMS ensemble inference combining outputs from multiple checkpoints.
- `example_mining.py` — Hard example mining utility for analyzing model predictions.

### Extension: All-Stage Feature Distillation (`knowledge-distillation-VGG16_AllStageFeature/`)

Extends ILMIL by distilling knowledge from **all intermediate feature stages** of the VGG16 backbone (stages 1–5), in addition to RPN and RCN features. This provides a richer distillation signal at the cost of slightly more computation.

**Key differences from `knowledge-distillation-VGG16/`:**
- `model/VGG.py` — Modified VGG16 that exposes features from each pooling stage separately.
- `model/faster_rcnn_vgg16.py` — `decom_vgg16()` returns intermediate features (`stage1`–`stage4`) and their corresponding pooling outputs, enabling multi-stage feature distillation in the trainer.

### Pruning: L1-Norm Channel Pruning (`simple-faster-rcnn-prune-VGG16/`)

Applies L1-norm based channel pruning to the VGG16 backbone of Faster R-CNN to reduce model size and computational cost. The pruning ratio is configurable per layer.

**Key components:**
- `faster-rcnn-prune_L1.py` — Main pruning script. Iterates over VGG16 convolutional layers, ranks channels by L1-norm of weights, and prunes the lowest-ranked channels.
- `model/faster_rcnn_vgg16_prune.py` — `FasterRCNNVGG16_PRUNE` class that supports dynamic channel configuration (per-layer channel counts stored in `cfg_mask`).
- `compute_flops.py` / `compute_flops_EC.py` — FLOPs computation utilities for evaluating model efficiency.
- `model_parameter.py` / `model_parameter_EC.py` — Model parameter counting utilities.

### Pruning + KD Combined (`knowledge-distillation-prune-VGG16/`)

Combines L1-norm channel pruning with MFRRK-based knowledge distillation. The workflow is:
1. **Prune** the pre-trained teacher model using L1-norm pruning to obtain a compact model.
2. **Distill** knowledge from the original (unpruned) teacher model to the pruned student model using MFRRK loss during incremental training on new classes.

This approach simultaneously achieves model compression and catastrophic forgetting mitigation.

**Key components:**
- Inherits the pruning framework from `simple-faster-rcnn-prune-VGG16/`.
- `trainer.py` — Extended trainer that incorporates both pruning masks (`cfg_mask`) and MFRRK distillation losses (`hint_loss`, `scores_loss`), distilling from the original teacher to the pruned student.
- `utils/cfg_mask.py` — Utilities for managing per-layer channel masks (`get_featureMap`, `clear_featureMap`, `get_mask_FM`, etc.) that control which channels are retained after pruning.

## Usage

### Environment Setup

```bash
# Install dependencies
pip install torch torchvision numpy scikit-image tqdm fire matplotlib visdom cython
pip install git+https://github.com/pytorch/tnt.git@master
```

### Data Preparation

The code uses the PASCAL VOC 2007 dataset format. Update `voc_data_dir` in `utils/config.py` to point to your local VOC dataset path before training:

```python
voc_data_dir = '/path/to/VOCdevkit2007/VOC2007'
```

### Incremental Learning (ILMIL)

```bash
# 1. Train the teacher model on old classes (baseline pre-training)
cd Faster-RCNN-incremental/knowledge-distillation-VGG16
python train.py

# 2. Modify VOC_BBOX_LABEL_NAMES in utils/config.py to include both old and new classes,
#    set load_path to the teacher checkpoint, and enable distillation flags.
#    Then train the student model with MFRRK distillation:
python train.py

# 3. Evaluate the trained model
python test.py
```

**Key configuration options in `utils/config.py`:**

| Option | Default | Description |
|---|---|---|
| `is_distillation` | `True` | Enable knowledge distillation mode |
| `use_hint` | `False` | Enable MFRRK intermediate feature distillation |
| `only_use_cls_distillation` | `False` | Use only classification-level distillation (ablation) |
| `load_path` | (see file) | Path to pre-trained teacher model checkpoint |
| `epoch` | `100` | Number of training epochs |
| `lr` | `1e-4` | Initial learning rate |
| `lr_decay` | `0.1` | Learning rate decay factor |
| `weight_decay` | `0.0005` | Weight decay for optimizer |
| `voc_data_dir` | (see file) | PASCAL VOC dataset root directory |

### Pruning + KD

```bash
# L1-norm pruning (standalone)
cd Faster-RCNN-prune/simple-faster-rcnn-prune-VGG16
python faster-rcnn-prune_L1.py

# Pruning + KD combined
cd Faster-RCNN-prune/knowledge-distillation-prune-VGG16
python train.py
```

### Ensemble Inference (Soft-NMS)

```bash
cd Faster-RCNN-incremental/knowledge-distillation-VGG16
python nms_Ensemble.py
```

### Hard Example Mining

```bash
cd Faster-RCNN-incremental/knowledge-distillation-VGG16
python example_mining.py
```

**Notes:**
- The default configuration paths in `utils/config.py` and some scripts are hardcoded to Linux paths. Please modify them to point to your local directories before running.
- A CUDA-capable GPU is required for training.
- The visdom server should be started before training for loss visualization: `python -m visdom.server`.

## Citation

If you find ILMIL useful or relevant to your project and research, please kindly cite our paper:

```
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
```

## Acknowledgement

This codebase is built upon the [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch) implementation. Special thanks to the authors for their open-source contribution.

## Contact

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
