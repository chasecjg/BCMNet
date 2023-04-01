# Bidirectional Collaborative Mentoring Network for Marine Organism Detection and Beyond

> **Authors:** 
> Jinguang Cheng,
> Zongwei Wu,
> Shuo Wang,
> Cédric Demonceaux,
> and Qiuping Jiang.

## 1. Preface

- This repository provides code for "_**Bidirectional Collaborative Mentoring Network for Marine Organism Detection and Beyond**_" TCSVT-2023.

## 2. Overview

### 2.1. Introduction
Organism detection plays a vital role in marine resource exploitation and the marine economy. How to accurately locate the target object within the camouflaged and dark light oceanic scene has recently drawn great attention in the research community. Existing learning-based works usually leverage local texture details within a neighboring area, with few methods explicitly exploring the usage of contextualized awareness for accurate object detection. From a novel perspective, we present a Bidirectional Collaborative Mentoring Network (BCMNet) which fully explores both texture and context clues during the encoding and decoding stages, making the cross-paradigm interaction bidirectional and improving the scene understanding at all stages. Specifically, we first extract texture and context features through a dual branch encoder and attentively fuse them through our adjacent feature fusion (AFF) block. Then, we propose a structure-aware module (SAM) and a detailenhanced module (DEM) to form our two-stage decoding pipeline. On the one hand, our SAM leverages both local and global clues to preserve morphological integrity and generate an initial prediction of the target object. On the other hand, the DEM explicitly explores long-range dependencies to refine the initially predicted object mask further. The combination of SAM and DEM enables better extracting, preserving, and enhancing the object morphology, making it easier to segment the target object from the camouflaged background with sharp contour. Extensive experiments on three benchmark datasets show that our proposed BCMNet performs favorably over state-of-the-art models.

### 2.2. Framework Overview

<p align="center">
    ![image](https://github.com/chasecjg/BCMNet/blob/main/Images/BCMNet.pdf)
    <em> 
    Figure 1: The overall architecture of the proposed model, which consists of two key components, i.e., attention-induced cross-level fusion module and dual-branch global context module. See § 3 in the paper for details.
    </em>
</p>

### 2.3. Qualitative Results

<p align="center">
    ![image](https://github.com/chasecjg/BCMNet/blob/main/Images/Compare_Results.pdf)
    <em> 
    Figure 2: Qualitative Results.
    </em>
</p>

## 3. Proposed Baseline

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
double 2080Ti GPU of 48 GB Memory.

1. Configuring your environment (Prerequisites):
    
    + Creating a virtual environment in terminal: `conda create -n C2FNet python=3.8`.
    
    + Installing necessary packages: `pip install -r requirements.txt`.

1. Downloading necessary data:

    + downloading training/testing dataset and move it into `./data/`, 
    which can be found in this [(Google Drive)](https://drive.google.com/file/d/1c0ToIqKMgaDyMT4YnS61toE0evAcnfck/view?usp=sharing).
    
    + downloading pretrained weights and move it into `./checkpoints/BCMNet.pth`, 
    which can be found in this [(Google Drive)](https://drive.google.com/file/d/1KZ53pNHXJXJma2vHHpFF7X5bwQcWK0kf/view?usp=sharing).
    
    + downloading ResNet weights and move it into `./models/res2net50_v1b_26w_4s-3cf99910.pth`[(Google Drive)](https://drive.google.com/file/d/1ITW3_ZBBv2JTviskxO9zfiqlaQ9Nlj-J/view?usp=sharing).

1. Training Configuration:

    + Assigning your costumed path, like `--train_save` and `--train_path` in `MyTrain.py`.
    + I modify the total epochs and the learning rate decay method (lib/utils.py has been updated), so there are differences from the training setup reported in the paper. Under the new settings, the training performance is more stable.

1. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `MyTest.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).

### 3.2 Evaluating your trained model:

One-key evaluation is written in python code (revised from [link](https://github.com/lartpang/PySODMetrics)), 

If you want to speed up the evaluation on GPU, you just need to use the efficient tool [link](https://github.com/lartpang/PySODMetrics) by `pip install pysodmetrics`.

Assigning your costumed path, like `method`, `mask_root` and `pred_root` in `eval.py`.

Just run `eval.py` to evaluate the trained model.



## 4. Citation


**[⬆ back to top](#1-preface)**
