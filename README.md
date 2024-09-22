<h1 align="center"> Predicting Gradient is Better: Exploring Self-Supervised Learning for SAR ATR with a Joint-Embedding Predictive Architecture </h1> 

<h5 align="center"><em> Weijie Li (李玮杰), Wei Yang (杨威), Tianpeng Liu (刘天鹏), Yuenan Hou (侯跃南), Yuxuan Li (李宇轩), Yongxiang Liu (刘永祥), and Li Liu (刘丽) </em></h5>

<p align="center">
<a href="https://arxiv.org/abs/2311.15153"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>
</p>

## Introduction

In the process of being updated (2024.9.22~2024.9.30) to enhance readability and usefulness.

正在更新中（2024.9.22~2024.9.30）以增强可读性和实用性

These are codes and weights of the paper：

 [Predicting Gradient is Better: Exploring Self-Supervised Learning for SAR ATR with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2311.15153):

百度网盘: 链接：https://pan.baidu.com/s/14sRPSCygTKMelSy4ZkqRzw?pwd=jeq8 提取码：jeq8

## Dataset

Dataset   | Size   | #Target | #Scene | Res(m)     | Band | Polarization | Description              
:-----------------------------:|:------:|:---------:|:--------:|:------------:|:----:|:------------:|:--------------------------------------------:
 [MSAR](https://radars.ac.cn/web/data/getData?dataType=MSAR)                          | 28,499 | >4   | >6  | 1           | C    | Quad         | Ground and sea target detection dataset      
 [SAR-Ship](https://github.com/CAESAR-Radi/SAR-Ship-Dataset)                      | 39,729 | >1   | >4  | 3~25    | C    | Quad         | Ship detection dataset in complex scenes     
 [SARSim](https://ieeexplore.ieee.org/abstract/document/7968358/)                 | 21,168 | 7         | 3        | 0.3          | X    | Single       | Vehicle simulation dataset                   
 [SAMPLE](https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public)                        | 5,380  | 10        | 1        | 0.3          | X    | Single       | Vehicle simulation and measured~dataset      
 [MSTAR](https://www.sdms.afrl.af.mil/index.php?collection=mstar)      | 5,216  | 10        | 1        | 0.3          | X    | Single       | Fine-grained vehicle classification dataset  
 [FUSAR-Ship](https://ieeexplore.ieee.org/abstract/document/9893301) | 9,830  | 10        | >5  | 1.1~1.7 | C    | Double       | Fine-grained ship classification dataset     
 [SAR-ACD](https://github.com/AICyberTeam/SAR-ACD)    | 2,537  | 6         | 3        | 1            | C    | Single       | Fine-grained aircraft classification dataset 


## Pre-training

Our code is based on [LoMaR](https://github.com/junchen14/LoMaR) with [MAE](https://github.com/facebookresearch/mae) and [MaskFeat](https://github.com/open-mmlab/mmselfsup/blob/0.x/configs/selfsup/maskfeat/README.md), and its enviroment is follow LoMaR.

* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

* The relative position encoding is modeled by following [iRPE](https://github.com/microsoft/Cream/tree/main/iRPE). To enable the iRPE with CUDA supported. Of curese, irpe can run without build.  

```
cd rpe_ops/
python setup.py install --user
```

For pre-training with default setting
```
CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node=4 --master_port=25642  main_pretrain.py --data_path ${IMAGENET_DIR}
```
Our main changes are in the model_lomar.py
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
## Fine-tuning with pre-trained checkpoints

Our few-shot learning is based on [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch). You may need to installate this and use our modified tools.py and transforms.py for SAR images. You can run MIM_finetune.sh and MIM_linear.sh.


## Acknowledgement

Many thanks to the research [LoMaR](https://github.com/junchen14/LoMaR), [MaskFeat](https://github.com/facebookresearch/SlowFast/blob/main/projects/maskfeat/README.md), [MAE](https://github.com/facebookresearch/mae), [FG-MAE](https://github.com/zhu-xlab/FGMAE), and [Dassl](https://github.com/zhu-xlab/FGMAE).

## Statement

This project is strictly forbidden for any commercial purpose. Any questions please contact us at lwj2150508321@sina.com. 
If you find our work is useful, please give us 🌟 in GitHub and cite our paper in the following BibTex format:

```
@article{li2023predicting,
  title={Predicting Gradient is Better: Exploring Self-Supervised Learning for {SAR} {ATR} with a Joint-Embedding Predictive Architecture },
  author={Li, Weijie and Wei, Yang and Liu, Tianpeng and Hou, Yuenan and Liu, Yongxiang and Liu, Li},
  journal={arXiv preprint},
  url={https://arxiv.org/abs/2311.15153},
  year={2024}
}
```
