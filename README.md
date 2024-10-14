<h1 align="center"> Predicting Gradient is Better: Exploring Self-Supervised Learning for SAR ATR with a Joint-Embedding Predictive Architecture </h1> 

<h5 align="center"><em> Weijie Li (李玮杰), Wei Yang (杨威), Tianpeng Liu (刘天鹏), Yuenan Hou (侯跃南), Yuxuan Li (李宇轩), Yongxiang Liu (刘永祥), and Li Liu (刘丽) </em></h5>

<p align="center">
  <a href="#Introduction">Introduction</a> |
  <a href="#Dataset">Dataset</a> |
  <a href="#Pre-training">Pre-training</a> |
  <a href="#Fine-tuning with pre-trained checkpoints">Fine-tuning</a> |
  <a href="#Acknowledgement">Acknowledgement</a> |
  <a href="#Statement">Statement</a>
</p >

<p align="center">
<a href="https://www.sciencedirect.com/science/article/pii/S0924271624003514"><img src="https://img.shields.io/badge/Paper-ISPRS%20Journal-orange"></a>
<a href="https://arxiv.org/abs/2311.15153"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>
<a href="https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8"><img src="https://img.shields.io/badge/Checkpoint-BaiduYun-blue"></a>
<a href="https://www.kaggle.com/models/liweijie19/sar-jepa"><img src="https://img.shields.io/badge/Checkpoint-Kaggle-blue"></a>  
</p>

If you find our work is useful, please give us 🌟 in GitHub and cite our paper in the following BibTex format:

```
@article{li2024predicting,
  title = {Predicting gradient is better: Exploring self-supervised learning for SAR ATR with a joint-embedding predictive architecture},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {218},
  pages = {326-338},
  year = {2024},
  issn = {0924-2716},
  doi = {https://doi.org/10.1016/j.isprsjprs.2024.09.013},
  url = {https://www.sciencedirect.com/science/article/pii/S0924271624003514},
  author = {Li, Weijie and Yang, Wei and Liu, Tianpeng and Hou, Yuenan and Li, Yuxuan and Liu, Zhen and Liu, Yongxiang and Liu, Li},
}
```

## Introduction

This is the official repository for the paper “Predicting Gradient is Better: Exploring Self-Supervised Learning for SAR ATR with a Joint-Embedding Predictive Architecture”, and here is our share link in [ISPRS](https://www.sciencedirect.com/science/article/pii/S0924271624003514?dgcid=author).

这里是论文 “Predicting Gradient is Better: Exploring Self-Supervised Learning for SAR ATR with a Joint-Embedding Predictive Architecture (预测梯度会更好：利用联合编码预测架构探索SAR ATR的自监督学习) ”的代码库，而论文的分享链接为[ISPRS](https://www.sciencedirect.com/science/article/pii/S0924271624003514?dgcid=author).

<figure>
<div align="center">
<img src=example/fig_framework.png width="90%">
</div>
</figure>

**Abstract:** The growing Synthetic Aperture Radar (SAR) data can build a foundation model using self-supervised learning (SSL) methods, which can achieve various SAR automatic target recognition (ATR) tasks with pretraining in large-scale unlabeled data and fine-tuning in small-labeled samples. SSL aims to construct supervision signals directly from the data, minimizing the need for expensive expert annotation and maximizing the use of the expanding data pool for a foundational model. This study investigates an effective SSL method for SAR ATR, which can pave the way for a foundation model in SAR ATR. The primary obstacles faced in SSL for SAR ATR are small targets in remote sensing and speckle noise in SAR images, corresponding to the SSL approach and signals. To overcome these challenges, we present a novel joint-embedding predictive architecture for SAR ATR (SAR-JEPA) that leverages local masked patches to predict the multi-scale SAR gradient representations of an unseen context. The key aspect of SAR-JEPA is integrating SAR domain features to ensure high-quality self-supervised signals as target features. In addition, we employ local masks and multi-scale features to accommodate various small targets in remote sensing. By fine-tuning and evaluating our framework on three target recognition datasets (vehicle, ship, and aircraft) with four other datasets as pretraining, we demonstrate its outperformance over other SSL methods and its effectiveness as the SAR data increases. This study demonstrates the potential of SSL for the recognition of SAR targets across diverse targets, scenes, and sensors. 

**摘要：** 可以基于不断增长的合成孔径雷达（SAR）数据和自监督学习（SSL）方法建立基础模型，通过在大规模无标注数据中进行预训练和在小标注样本中进行微调，实现各种 SAR 自动目标识别（ATR）任务。SSL 旨在直接从数据中构建监督信号，最大限度地减少对昂贵的专家标注的需求，并且利用不断扩大的数据建立基础模型。本研究探讨了一种适用于 SAR ATR 的 SSL 方法，它可以为 SAR ATR 的基础模型铺平道路。用于 SAR ATR 的 SSL 所面临的主要障碍是遥感中的小目标和 SAR 图像中的散斑噪声，与 SSL 方法和信号相对应。为了克服这些挑战，我们提出了一种用于 SAR ATR 的新型联合编码预测架构（SAR-JEPA），该架构利用局部掩码块来预测未见背景的多尺度 SAR 梯度表示。SAR-JEPA 的关键在于结合 SAR 图像域特征，确保将高质量的自监督信号作为目标特征。此外，我们还采用了局部掩码和多尺度特征，以适应遥感中的各种小型目标。通过在四个SAR目标数据集上进行无标签预训练，以及其他三个目标识别数据集（车辆、船舶和飞机）上对我们的框架进行微调和评估，我们证明了它优于其他 SSL 方法的性能，以及随着SAR数据的增加其有效性。这项研究表明了 SSL 在识别不同目标、场景和传感器的合成孔径雷达目标方面的潜力。

## Dataset

Dataset   | Size   | #Target | #Scene | Res(m)     | Band | Polarization | Description              
:-----------------------------:|:------:|:---------:|:--------:|:------------:|:----:|:------------:|:--------------------------------------------:
 [MSAR](https://radars.ac.cn/web/data/getData?dataType=MSAR)                          | 28,499 | >4   | >6  | 1           | C    | Quad         | Ground and sea target detection dataset      
 [SAR-Ship](https://github.com/CAESAR-Radi/SAR-Ship-Dataset)                      | 39,729 | >1   | >4  | 3~25    | C    | Quad         | Ship detection dataset in complex scenes     
 [SARSim](https://ieeexplore.ieee.org/abstract/document/7968358/)                 | 21,168 | 7         | 3        | 0.3          | X    | Single       | Vehicle simulation dataset                   
 [SAMPLE](https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public)                        | 5,380  | 10        | 1        | 0.3          | X    | Single       | Vehicle simulation and measured~dataset      
 [MSTAR](https://www.sdms.afrl.af.mil/index.php?collection=mstar)      | 5,216  | 10        | 1        | 0.3          | X    | Single       | Fine-grained vehicle classification dataset  
 [FUSAR-Ship](https://link.springer.com/article/10.1007/s11432-019-2772-5) | 9,830  | 10        | >5  | 1.1~1.7 | C    | Double       | Fine-grained ship classification dataset     
 [SAR-ACD](https://github.com/AICyberTeam/SAR-ACD)    | 2,537  | 6         | 3        | 1            | C    | Single       | Fine-grained aircraft classification dataset 

You can cite the above dataset paper and download the required datasets directly from our [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8). 
The first four datasets are used as a pre-training set, and the next three are vehicle, ship, and aircraft datasets for fine-tuning the classification task. 
The MSTAR setting follows our previous [paper](https://ieeexplore.ieee.org/document/10283916).
The FUSAR-Ship setting follows the [paper](https://ieeexplore.ieee.org/abstract/document/9893301).
The SAR-ACD is randomly split as training and test sets with 5 type (A220, A330, ARJ21, Boeing737, and Boeing787).

您可以引用上述数据集论文，并从我们的 [百度云](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8) 下载所需的数据集。
前四个数据集作为预训练集，后面三个分别为车辆、舰船和飞机数据集用于分类任务微调。
MSTAR 设置沿用了我们之前的[论文](https://ieeexplore.ieee.org/document/10283916)。
FUSAR-Ship 数据集的设置遵循[论文](https://ieeexplore.ieee.org/abstract/document/9893301)。
SAR-ACD 设置为随机切分作为训练集和测试集，并且使用了5种型号（A220、A330、ARJ21、Boeing737和Boeing787）。

## Pre-training

Our code enviroment follows LoMaR.

* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

* The relative position encoding is modeled by following [iRPE](https://github.com/microsoft/Cream/tree/main/iRPE). To enable the iRPE with CUDA supported. Of curese, irpe can run without build and slower speed.  

```
cd rpe_ops/
python setup.py install --user
```

For pre-training with default setting
```
CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node=4 --master_port=25642  main_pretrain.py --data_path ${IMAGENET_DIR}
```
Our main changes are in the [model_lomar.py](https://github.com/waterdisappear/SAR-JEPA/blob/main/Pretraining/models_lomar.py)
```
        self.sarfeature1 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=5,
                                  img_size=self.img_size,patch_size=self.patch_size)
        self.sarfeature2 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=9,
                                  img_size=self.img_size,patch_size=self.patch_size)
        self.sarfeature3 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=13,
                                  img_size=self.img_size,patch_size=self.patch_size)
        self.sarfeature4 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=17,
                                  img_size=self.img_size,patch_size=self.patch_size)
```

Here are the pre-training weights we obtained using different methods.

以下是我们利用不同方法得到的预训练权重。

Method   | Downlaod   | Method   | Downlaod        
:-----------------:|:-----------------:|:-----------------:|:-----------------:
[MAE](https://github.com/facebookresearch/mae) | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FMAE&parentPath=%2F) | [LoMaR](https://github.com/junchen14/LoMaR) | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FLoMaR&parentPath=%2F)
[I-JEPA](https://github.com/facebookresearch/ijepa) | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2Fijepa&parentPath=%2F) | [FG-MAE](https://github.com/zhu-xlab/FGMAE) | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FFG-MAE&parentPath=%2F)
[LoMaR-SAR](https://www.sciencedirect.com/science/article/pii/S0924271624003514) | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FLoMaR-SAR&parentPath=%2F) | [low pass filter](https://github.com/facebookresearch/SlowFast/blob/main/projects/maskfeat/README.md) | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FLoMaR_low_pass_filter&parentPath=%2F)
[SAR HOG](https://www.mdpi.com/2072-4292/8/8/683) | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FSAR%20HOG&parentPath=%2F) | [PGGA](https://ieeexplore.ieee.org/document/10035918) | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FPGGA&parentPath=%2F)
[SAR-JEPA (Gaussion keneral)](https://www.sciencedirect.com/science/article/pii/S0924271624003514) | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FSAR-JEPA_Gaussion%20keneral&parentPath=%2F) | [SAR-JEPA](https://www.sciencedirect.com/science/article/pii/S0924271624003514) | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FSAR-JEPA&parentPath=%2F)

## Fine-tuning with pre-trained checkpoints

Our few-shot learning is based on [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch). You need to install this and use our modified ''Dassl.pytorch\dassl\utils\tools.py'' and ''Dassl.pytorch\dassl\data\transforms\transforms.py'' in our modified [zip](https://github.com/waterdisappear/SAR-JEPA/blob/main/few_shot_classification/Dassl.pytorch.zip) for SAR single-channel amplitude images. Then, you can run our [MIM_finetune.sh](https://github.com/waterdisappear/SAR-JEPA/blob/main/few_shot_classification/finetune/MIM_finetune.sh) and [MIM_linear.sh](https://github.com/waterdisappear/SAR-JEPA/blob/main/few_shot_classification/finetune/MIM_linear.sh) for evaluations.

我们的 few-shot 学习基于 [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch)。您需要安装并使用我们为 SAR 单通道振幅图像修改过的[zip](https://github.com/waterdisappear/SAR-JEPA/blob/main/few_shot_classification/Dassl.pytorch.zip)中的''Dassl.pytorch\dassl\utils\tools.py'' 和''Dassl.pytorch\dassl\data\transforms\transforms.py'' 。然后，您可以运行我们的 [MIM_finetune.sh](https://github.com/waterdisappear/SAR-JEPA/blob/main/few_shot_classification/finetune/MIM_finetune.sh) 和 [MIM_linear.sh](https://github.com/waterdisappear/SAR-JEPA/blob/main/few_shot_classification/finetune/MIM_linear.sh) 进行评估。

<figure>
<div align="center">
<img src=example/fig_training_curve.png width="50%">
</div>
</figure>

You can visualise the attention distance using the code in [plt_attention_distance](plt_attention_distance)

你可以利用[plt_attention_distance](plt_attention_distance)中的代码进行注意力距离的可视化

<figure>
<div align="center">
<img src=example/attention_distance.png width="80%">
</div>
</figure>


## Acknowledgement

We extend our deepest gratitude to research ([LoMaR](https://github.com/junchen14/LoMaR), [MaskFeat](https://github.com/facebookresearch/SlowFast/blob/main/projects/maskfeat/README.md), [MAE](https://github.com/facebookresearch/mae), [I-JEPA](https://github.com/facebookresearch/ijepa), [FG-MAE](https://github.com/zhu-xlab/FGMAE), and [Dassl](https://github.com/zhu-xlab/FGMAE)) and pubilc SAR datasets ([MSAR](https://radars.ac.cn/web/data/getData?dataType=MSAR), [SAR-Ship](https://github.com/CAESAR-Radi/SAR-Ship-Dataset), [SARSim](https://ieeexplore.ieee.org/abstract/document/7968358/), [SAMPLE](https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public), [MSTAR](https://www.sdms.afrl.af.mil/index.php?collection=mstar),  [FUSAR-Ship](https://ieeexplore.ieee.org/abstract/document/9893301), and [SAR-ACD](https://github.com/AICyberTeam/SAR-ACD)).  Their selfless contributions and dedication have greatly facilitated and promoted research in this field.

我们衷心感谢相关研究（[LoMaR](https://github.com/junchen14/LoMaR)、[MaskFeat](https://github.com/facebookresearch/SlowFast/blob/main/projects/maskfeat/README.md)、[MAE](https://github.com/facebookresearch/mae)、[I-JEPA](https://github.com/facebookresearch/ijepa)、[FG-MAE](https://github.com/zhu-xlab/FGMAE)和[Dassl](https://github.com/zhu-xlab/FGMAE)）和公开的合成孔径雷达数据集（[MSAR](https://radars.ac.cn/web/data/getData?dataType=MSAR)、[SAR-Ship](https://github.com/CAESAR-Radi/SAR-Ship-Dataset)、[SARSim](https://ieeexplore.ieee.org/abstract/document/7968358/)、[SAMPLE](https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public)、[MSTAR](https://www.sdms.afrl.af.mil/index.php?collection=mstar)、[FUSAR-Ship](https://ieeexplore.ieee.org/abstract/document/9893301) 和 [SAR-ACD](https://github.com/AICyberTeam/SAR-ACD))。 他们的无私贡献极大地促进和推动了这一领域的研究。

## Statement

- This project is released under the [CC BY-NC 4.0](LICENSE).
- Any questions please contact us at lwj2150508321@sina.com. 


