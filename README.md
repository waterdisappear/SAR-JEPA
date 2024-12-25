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
<a href="https://zhuanlan.zhihu.com/p/787495052"><img src="https://img.shields.io/badge/文章-知乎-blue"></a>    
</p>




## Introduction

This is the official repository for the paper “Predicting Gradient is Better: Exploring Self-Supervised Learning for SAR ATR with a Joint-Embedding Predictive Architecture”, and here is our share link in [ISPRS](https://www.sciencedirect.com/science/article/pii/S0924271624003514?dgcid=author). Inspired by [JEPA](https://arxiv.org/abs/2301.08243), we center on self-supervised learning in a special feature rather than the original pixel space. This change is very effective in SAR images, where pixel values are disturbed by speckle noise. Besides, information compression is realized for the original pixels, improving learning efficiency. After introducing JEPA to Earth observation, [AnySat](https://github.com/gastruc/AnySat/blob/main/README.md) use the JEPA architecture in multimodality to extract common semantic information. 

<figure>
<div align="center">
<img src=example/fig_framework.png width="90%">
</div>
</figure>


**Abstract:** The growing Synthetic Aperture Radar (SAR) data can build a foundation model using self-supervised learning (SSL) methods, which can achieve various SAR automatic target recognition (ATR) tasks with pretraining in large-scale unlabeled data and fine-tuning in small-labeled samples. SSL aims to construct supervision signals directly from the data, minimizing the need for expensive expert annotation and maximizing the use of the expanding data pool for a foundational model. This study investigates an effective SSL method for SAR ATR, which can pave the way for a foundation model in SAR ATR. The primary obstacles faced in SSL for SAR ATR are small targets in remote sensing and speckle noise in SAR images, corresponding to the SSL approach and signals. To overcome these challenges, we present a novel joint-embedding predictive architecture for SAR ATR (SAR-JEPA) that leverages local masked patches to predict the multi-scale SAR gradient representations of an unseen context. The key aspect of SAR-JEPA is integrating SAR domain features to ensure high-quality self-supervised signals as target features. In addition, we employ local masks and multi-scale features to accommodate various small targets in remote sensing. By fine-tuning and evaluating our framework on three target recognition datasets (vehicle, ship, and aircraft) with four other datasets as pretraining, we demonstrate its outperformance over other SSL methods and its effectiveness as the SAR data increases. This study demonstrates the potential of SSL for the recognition of SAR targets across diverse targets, scenes, and sensors. 

## Dataset

|                           Dataset                            |  Size  | #Target | #Scene | Res(m)  | Band | Polarization |                 Description                  |
| :----------------------------------------------------------: | :----: | :-----: | :----: | :-----: | :--: | :----------: | :------------------------------------------: |
| [MSAR](https://radars.ac.cn/web/data/getData?dataType=MSAR)  | 28,499 |   >4    |   >6   |    1    |  C   |     Quad     |   Ground and sea target detection dataset    |
| [SAR-Ship](https://github.com/CAESAR-Radi/SAR-Ship-Dataset)  | 39,729 |   >1    |   >4   |  3~25   |  C   |     Quad     |   Ship detection dataset in complex scenes   |
| [SARSim](https://ieeexplore.ieee.org/abstract/document/7968358/) | 21,168 |    7    |   3    |   0.3   |  X   |    Single    |          Vehicle simulation dataset          |
| [SAMPLE](https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public) | 5,380  |   10    |   1    |   0.3   |  X   |    Single    |   Vehicle simulation and measured~dataset    |
| [MSTAR](https://www.sdms.afrl.af.mil/index.php?collection=mstar) | 5,216  |   10    |   1    |   0.3   |  X   |    Single    | Fine-grained vehicle classification dataset  |
| [FUSAR-Ship](https://link.springer.com/article/10.1007/s11432-019-2772-5) | 9,830  |   10    |   >5   | 1.1~1.7 |  C   |    Double    |   Fine-grained ship classification dataset   |
|      [SAR-ACD](https://github.com/AICyberTeam/SAR-ACD)       | 2,537  |    6    |   3    |    1    |  C   |    Single    | Fine-grained aircraft classification dataset |

You can cite the above dataset paper and download the required datasets directly from our [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8). 
The first four datasets are used as a pre-training set, and the next three are vehicle, ship, and aircraft datasets for fine-tuning the classification task. 
The MSTAR setting follows our previous [paper](https://ieeexplore.ieee.org/document/10283916).
The FUSAR-Ship setting follows the [paper](https://ieeexplore.ieee.org/abstract/document/9893301).
The SAR-ACD is randomly split as training and test sets with 5 type (A220, A330, ARJ21, Boeing737, and Boeing787).

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

|                            Method                            |                           Downlaod                           |                            Method                            |                           Downlaod                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|        [MAE](https://github.com/facebookresearch/mae)        | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FMAE&parentPath=%2F) |         [LoMaR](https://github.com/junchen14/LoMaR)          | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FLoMaR&parentPath=%2F) |
|     [I-JEPA](https://github.com/facebookresearch/ijepa)      | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2Fijepa&parentPath=%2F) |         [FG-MAE](https://github.com/zhu-xlab/FGMAE)          | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FFG-MAE&parentPath=%2F) |
| [LoMaR-SAR](https://www.sciencedirect.com/science/article/pii/S0924271624003514) | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FLoMaR-SAR&parentPath=%2F) | [low pass filter](https://github.com/facebookresearch/SlowFast/blob/main/projects/maskfeat/README.md) | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FLoMaR_low_pass_filter&parentPath=%2F) |
|      [SAR HOG](https://www.mdpi.com/2072-4292/8/8/683)       | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FSAR%20HOG&parentPath=%2F) |    [PGGA](https://ieeexplore.ieee.org/document/10035918)     | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FPGGA&parentPath=%2F) |
| [SAR-JEPA (Gaussion keneral)](https://www.sciencedirect.com/science/article/pii/S0924271624003514) | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FSAR-JEPA_Gaussion%20keneral&parentPath=%2F) | [SAR-JEPA](https://www.sciencedirect.com/science/article/pii/S0924271624003514) | [BaiduYun](https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8#list/path=%2FSAR-JEPA%2Fweights%2FSAR-JEPA&parentPath=%2F) |

## Fine-tuning with pre-trained checkpoints

Our few-shot learning is based on [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch). You need to install this and use our modified ''Dassl.pytorch\dassl\utils\tools.py'' and ''Dassl.pytorch\dassl\data\transforms\transforms.py'' in our modified [zip](https://github.com/waterdisappear/SAR-JEPA/blob/main/few_shot_classification/Dassl.pytorch.zip) for SAR single-channel amplitude images. Then, you can run our [MIM_finetune.sh](https://github.com/waterdisappear/SAR-JEPA/blob/main/few_shot_classification/finetune/MIM_finetune.sh) and [MIM_linear.sh](https://github.com/waterdisappear/SAR-JEPA/blob/main/few_shot_classification/finetune/MIM_linear.sh) for evaluations.

<figure>
<div align="center">
<img src=example/fig_training_curve.png width="50%">
</div>
</figure>


You can visualise the attention distance using the code in [plt_attention_distance](plt_attention_distance)

<figure>
<div align="center">
<img src=example/attention_distance.png width="80%">
</div>
</figure>



## Acknowledgement

We extend our deepest gratitude to research ([LoMaR](https://github.com/junchen14/LoMaR), [MaskFeat](https://github.com/facebookresearch/SlowFast/blob/main/projects/maskfeat/README.md), [MAE](https://github.com/facebookresearch/mae), [I-JEPA](https://github.com/facebookresearch/ijepa), [FG-MAE](https://github.com/zhu-xlab/FGMAE), and [Dassl](https://github.com/zhu-xlab/FGMAE)) and pubilc SAR datasets ([MSAR](https://radars.ac.cn/web/data/getData?dataType=MSAR), [SAR-Ship](https://github.com/CAESAR-Radi/SAR-Ship-Dataset), [SARSim](https://ieeexplore.ieee.org/abstract/document/7968358/), [SAMPLE](https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public), [MSTAR](https://www.sdms.afrl.af.mil/index.php?collection=mstar),  [FUSAR-Ship](https://ieeexplore.ieee.org/abstract/document/9893301), and [SAR-ACD](https://github.com/AICyberTeam/SAR-ACD)).  Their selfless contributions and dedication have greatly facilitated and promoted research in this field.

## Statement

- This project is released under the [CC BY-NC 4.0](LICENSE).
- Any questions please contact us at lwj2150508321@sina.com. 
- If you find our work is useful, please give us 🌟 in GitHub and cite our paper in the following BibTex format:

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

@article{li2024saratr,
  title={SARATR-X: Towards Building A Foundation Model for SAR Target Recognition},
  author={Li, Weijie and Yang, Wei and Hou, Yuenan and Liu, Li and Liu, Yongxiang and Li, Xiang},
  journal={arXiv preprint},
  url={https://arxiv.org/abs/2405.09365},
  year={2024}
}
```
