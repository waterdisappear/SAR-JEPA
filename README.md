# SAR-JEPA: A Joint-Embedding Predictive Architecture for SAR ATR

I'll release codes and weight as soon as I'm done other projects, no later than April of this year.

These are codes and weights of the paper [Predicting Gradient is Better: Exploring Self-Supervised Learning for SAR ATR with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2311.15153):

## Dataset

Dataset   | Size   | \# Target | \# Scene | Res. (m)     | Band | Polarization | Description              
:-----------------------------:|:------:|:---------:|:--------:|:------------:|:----:|:------------:|:--------------------------------------------:
 MSAR                          | 28,499 | $\geq$4   | $\geq$6  | 1~           | C    | Quad         | Ground and sea target detection dataset      
 SAR-Ship                      | 39,729 | $\geq$1   | $\geq$4  | 3$\sim$25    | C    | Quad         | Ship detection dataset in complex scenes     
 SARSim                        | 21,168 | 7         | 3        | 0.3          | X    | Single       | Vehicle simulation dataset                   
 SAMPLE                        | 5,380  | 10        | 1        | 0.3          | X    | Single       | Vehicle simulation and measured~dataset      
 \textcolor{black}{MSTAR}      | 5,216  | 10        | 1        | 0.3          | X    | Single       | Fine-grained vehicle classification dataset  
 \textcolor{black}{FUSAR-Ship} | 9,830  | 10        | $\geq$5  | 1.1$\sim$1.7 | C    | Double       | Fine-grained ship classification dataset     
 \textcolor{black}{SAR-ACD}    | 2,537  | 6         | 3        | 1            | C    | Single       | Fine-grained aircraft classification dataset 



[MSAR](https://radars.ac.cn/web/data/getData?dataType=MSAR), [SAR-Ship](https://github.com/CAESAR-Radi/SAR-Ship-Dataset), SARSim, SAMPLE, MSTAR, FUSAR-Ship，SAR-ACD
Google Drive:
百度网盘:

## Pre-training


## Fine-tuning with pre-trained checkpoints


## Weights
Lists

Google Drive:
百度网盘:

## Contact us
If you have any questions, please contact us at lwj2150508321@sina.com
