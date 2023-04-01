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

<p>
  <img src="https://github.com/chasecjg/BCMNet/blob/main/Images/BCMNet.png" alt="BCMNet Architecture">
  <br>
  <em>Figure 1: Architecture Overview. Our BCMNet consists of a dual-branch encoder (Sec. III-B), feature fusion module AFF (Sec. III-C), an initial decoder based on SAM (Sec. III-D), and a refiner based on DEM (Sec. III-E). During feature extraction, each encoder has its specific purposes, i.e., the texture encoder analyzes local details based on colour information, while the context encoder leverages the long-range dependencies for contextualized awareness. To fuse the encoded features, we propose an AFF module to aggregate features from different paradigms and scales. Then we introduce the SAM to partially decode the feature and generate the initial prediction based on the object's external shape. Finally, the partial mask is further refined by enhancing the awareness of local fine-grained details through DEM. The initial and final predictions are both supervised by the ground truth mask, making our network end-to-end trainable.</em>
</p>



### 2.3. Qualitative Results

<p>
    <img src="https://github.com/chasecjg/BCMNet/blob/main/Images/Compare_Results.png">
    <br>
    <em> 
    Figure 2: Qualitative Results.
    </em>
</p>

### 2.4. Qualitative Results

\begin{table*}[!t]
\caption{Comparison with SOTA models on MAS3K, CHAMELEON, and COD10K datasets. The best top three results are highlighted in \textcolor{red}{red}, \textcolor{blue}{blue}, and \textcolor{green}{green}, respectively. All methods are trained/tested on the same images as ours.  We evaluate the metric with structural similarity ($S_{\alpha}$ \cite{39}),  weighted F-measure ($F_{\beta}^{\omega}$ \cite{40}), 
mean absolute error ($M$ \cite{38}), and mean E-measure (m$E_{\varphi}$ \cite{109}).
$\uparrow$ means higher scores are better, while $\downarrow$ means lower scores are better. The * represents medical image segmentation.}
\vspace{-3mm}
\label{tab:Comparison}
\centering
\renewcommand\arraystretch{1.2}
\resizebox{\textwidth}{!}
{$
\begin{tabular}{cc|cccc|cccc|cccc}
\toprule
 \multirow{2}{*}{Models} &   \multirow{2}{*}{Publication}     & \multicolumn{4}{c|}{MAS3K \cite{56} }                                                                                               & \multicolumn{4}{c|}{CHAMELEON \cite{23}  }                                                  & \multicolumn{4}{c}{COD10K \cite{13}}                                                                                          \\
\cline{3-14}
 &  & $\mathrm{S} \alpha \uparrow$                          & $F_{\beta}^{\omega}\uparrow$                             & $M\downarrow$                         & $mE_{\varphi}\uparrow$                         & $\mathrm{S} \alpha \uparrow$                          & $F_{\beta}^{\omega}\uparrow$                             & $M\downarrow$                         & $mE_{\varphi}\uparrow$                         & $\mathrm{S} \alpha \uparrow$                          & $F_{\beta}^{\omega}\uparrow$                             & $M\downarrow$                         & $mE_{\varphi}\uparrow$                         \\
\midrule
\multicolumn{14}{c}{Methods Initially Designed for Salient Object Detection / Medical Image Segmentation} \\
\midrule
CPD \cite{46}     & 2019 CVPR            & 0.869                        & 0.776                        & 0.032                        & 0.897                        & 0.870                        & 0.758                        & 0.034                        & 0.886                        & 0.774                        & 0.588                        & 0.041                        & 0.801                        \\
\midrule
SCRN \cite{47}     &2019 ICCV         & 0.870                        & 0.750                        & 0.033                        & 0.890                        & 0.865                        & 0.722                        & 0.043                        & 0.871                        & 0.789                        & 0.572                        & 0.046                        & 0.801                        \\
\midrule
F3Net \cite{48}      &2020 AAAI        & 0.872                        & 0.801                        & 0.028                        & 0.927                        & 0.868                        & 0.764                        & 0.038                        & 0.917                        & 0.805                        & 0.650                        & 0.039                        & 0.872                        \\
\midrule
PraNet* \cite{49}   &2020 MICCAI        & 0.883                        & 0.817                        & 0.026                        & 0.929                        & 0.873                        & 0.785                        & 0.034                        & 0.922                        & 0.812                        & 0.671                        & 0.036                        & 0.877                        \\
\midrule
PolyP-Pvt* \cite{50}    &2021 TMI        & 0.889                        & 0.840                        & 0.027                        & 0.934                        & 0.881                        & 0.830                        & 0.026                        & 0.943                        & 0.814                        & 0.705                        & 0.035                        & 0.887                        \\
\midrule
PFNet \cite{mei2021camouflaged}      &2021 CVPR        & 0.882                        & 0.818                        & 0.026                        & 0.927                        & 0.840                        & 0.814                        & 0.028                        & 0.942                        & 0.798                        & 0.643                        & 0.037                        & 0.865                        \\
\midrule
PFSNet \cite{51}     &2021 AAAI         & 0.880                        & 0.816                        & 0.027                        & 0.929                        & 0.866                        & 0.781                        & 0.033                        & 0.924                        & 0.806                        & 0.669                        & 0.038                        & 0.876                        \\
\midrule
PSGLoss \cite{52}     &2021 TIP         & 0.848                        & 0.779                        & 0.031                        & 0.883                        & 0.828                        & 0.753                        & 0.031                        & 0.872                        & 0.732                        & 0.566                        & 0.040                        & 0.732                        \\
\midrule
\multicolumn{14}{c}{Methods Initially Designed for Camouflaged Object Detection}       \\
\midrule
SINet-v1 \cite{13}    &2020 CVPR      & 0.870                        & 0.766                        & 0.031                        & 0.902                        & 0.867                        & 0.741                        & 0.039                        & 0.891                        & 0.789                        & 0.595                        & 0.042                        & 0.819                        \\
\midrule
SINet-v2 \cite{53}   &2021 PAMI        & 0.894                        & 0.843                        & {\textcolor{blue}{0.021}} & 0.942                        & {\textcolor{blue}{0.897}} & 0.828                        & {\textcolor{green}{0.026}} & {\textcolor{blue}{0.950}} & {\textcolor{green}{0.829}}                        & 0.707                        &{\textcolor{green}{0.032}}                        & {\textcolor{blue}{0.899}} \\
\midrule
C2FNet-v1 \cite{14}  &2021 IJCAI      & {\textcolor{green}{0.897}} & {\textcolor{green}{0.850}} & {\textcolor{blue}{0.021}} & {\textcolor{blue}{0.943}} & {\textcolor{blue}{0.897}} & {\textcolor{blue}{0.840}} & {\textcolor{blue}{0.025}} & {\textcolor{green}{0.944}} & {\textcolor{blue}{0.830}} & {\textcolor{green}{0.714}} & 0.030                        & {\textcolor{red}{0.902}} \\
\midrule
RankNet \cite{54}    &2021 CVPR        & 0.858                        & 0.764                        & 0.034                        & 0.909                        & 0.838                        & 0.707                        & 0.048                        & 0.884                        & 0.782                        & 0.599                        & 0.049                        & 0.841                        \\
\midrule
BSANet \cite{26}    &2022 AAAI        & {\textcolor{blue}{0.900}} & {\textcolor{blue}{0.856}} & {\textcolor{blue}{0.021}} & {\textcolor{blue}{0.943}} & 0.888                        & 0.830                        & 0.027                        & 0.941                        & {\textcolor{red}{0.833}} & {\textcolor{blue}{0.722}} & \textcolor{blue}{0.028}                        & 0.897                        \\
\midrule
 C2FNet-v2 \cite{55}     &2022 TCSVT       & 0.898                        & 0.852                        & 0.022                        & 0.939                        & 0.891                        & {\textcolor{green}{0.839}} & 0.026 & 0.942                        & 0.827                        & 0.715                        & 0.031                        & 0.896                        \\
\midrule
 ECDNet \cite{56}     &2022 TCSVT      & 0.850                        & 0.766                        & 0.036                        & 0.901                        & 0.843                        & 0.749                        & 0.038                        & 0.893                        & 0.683                        & 0.446                        & 0.049                        & 0.781                        \\
\midrule
BCMNet             & Ours                   & {\textcolor{red}{0.906}} & {\textcolor{red}{0.865}} & {\textcolor{red}{0.019}} & {\textcolor{red}{0.945}} & {\textcolor{red}{0.900}} & {\textcolor{red}{0.863}} & {\textcolor{red}{0.022}} & {\textcolor{red}{0.954}} & {\textcolor{green}{0.829}} & {\textcolor{red}{0.723}} & {\textcolor{red}{0.027}} & {\textcolor{blue}{0.899}}                        \\
\bottomrule
\end{tabular}$}
\end{table*}
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
