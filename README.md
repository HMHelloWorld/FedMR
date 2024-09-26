The source code for **Is Aggregation the Only Choice? Federated Learning via Layer-wise Model Recombination** (Accepted by KDD2024)

https://dl.acm.org/doi/abs/10.1145/3637528.3671722


## 1. Environment setting requirements
* Python 3.7
* PyTorch

## 2. Instruction
### 2.1 Parameter
#### 2.1.1 Dataset Setting
`--dataset <dataset name>`
We can set ‘cifar10’, ‘cifar100’ and ‘femnist’ for CIFAR-10, CIFAR-100, and FEMNIST.
#### 2.1.2 Model Settings
`--num_classes <number>`
Set the number of classes Set 10 for CIFAR-10

Set 20 for CIFAR-100

Set 62 for FEMNIST

`--num_channels <number>`
Set the number of channels of data Set 3 for CIFAR-10 and CIFAR-100. Set 1 for FEMNIST.
#### 2.1.3 Data heterogeneity
`--iid <0 or 1>`

0 – set non-iid 1 – set iid

`--data_beta <𝛼>`

Set the 𝛂 for the Dirichlet distribution

`--generate_data <0 or 1>`

0 – use the existing configuration of 𝑫𝒊𝒓(𝜶) 1 – generate a new configuration of 𝑫𝒊𝒓(𝜶)

#### 2.1.2 FL Settings
`--epochs <number of rounds>`

Set the number of training rounds.

#### 2.1.2 FedMR and Baseline Settings
`-- algorithm <baseline name>`

Set the baseline name:
* FedMR
* FedAvg
* FedProx
* FedGen
* ClustererSampling

`-- first_stage_bound <num>`

Set the hyperparameter for FedMR

## 3. Citation
```
@inproceedings{hu2024aggregation,
    title={Is Aggregation the Only Choice? Federated Learning via Layer-wise Model Recombination},
    author={Hu, Ming and Yue, Zhihao and Xie, Xiaofei and Chen, Cheng and Huang, Yihao and Wei, Xian and Lian, Xiang and Liu, Yang and Chen, Mingsong},
    booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    pages={1096--1107},
    year={2024}
}
```
