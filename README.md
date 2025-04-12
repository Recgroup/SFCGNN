# SFCGNN: Mitigating Graph Neural Network Over-Smoothing from Structure and Feature Perspectives
Official pytorch code for paper [**SFCGNN: Mitigating Graph Neural Network Over-Smoothing from Structure and Feature Perspectives**]

## Introduction
Our code was experimented with under torch 2.0.0 mitigation, you can also run the code in other suitable environments.

## Examples
Here are some of the commands we provide that you can use to run this code on pycharm.
```Cora
python main.py --data cora --model SFCGNN --hid 128 --lr 0.01 --epochs 200 --wightDecay 0.0005 --nlayer 16 --seed 30  --tau 0.8 --dropout 0.6 --maskingRate 0.2 --aug_adj 1 --t 2
```
```Citeseer
python main.py --data citeseer --model SFCGNN --hid 128 --lr 0.1 --epochs 120 --wightDecay 0.0005 --nlayer 8 --seed 30  --tau 0.5 --dropout 0.8 --maskingRate 0.25 --aug_adj 1 --t 2
```
```Pubmed
python main.py --data pubmed --model SFCGNN --hid 128 --lr 0.05 --epochs 120 --wightDecay 0.0005 --nlayer 16 --seed 30  --tau 0.7 --dropout 0.5 --maskingRate 0.3 --aug_adj 1 --t 2
```




