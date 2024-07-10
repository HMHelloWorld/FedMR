# Federated Learning [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4321561.svg)](https://doi.org/10.5281/zenodo.4321561)

This is partly the reproduction of the paper of [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)   
Only experiments on MNIST and CIFAR10 (both IID and non-IID) is produced by far.

Note: The scripts will be slow without the implementation of parallel computing. 

## Requirements
python>=3.6  
pytorch>=0.4

## Run

The MLP and CNN models are produced by:
> python [main_nn.py](main_nn.py)

Federated learning with MLP and CNN is produced by:
> python [main_fed.py](main_fed.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  

NB: for CIFAR-10, `num_channels` must be 3.

## Results


## Ackonwledgements
Acknowledgements give to [youkaichao](https://github.com/youkaichao).

## References
McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Artificial Intelligence and Statistics (AISTATS), 2017.


Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of Machine Learning and Systems 2 (2020): 429-450.


Li, Qinbin, Bingsheng He, and Dawn Song. "Model-contrastive federated learning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

Fraboni, Yann, et al. "Clustered sampling: Low-variance and improved representativity for clients selection in federated learning." International Conference on Machine Learning. PMLR, 2021.

Yao, Dezhong, et al. "Local-Global Knowledge Distillation in Heterogeneous Federated Learning with Non-IID Data." arXiv preprint arXiv:2107.00051 (2021).

Zhu, Zhuangdi, Junyuan Hong, and Jiayu Zhou. "Data-free knowledge distillation for heterogeneous federated learning." International Conference on Machine Learning. PMLR, 2021.

Gao, Liang, et al. "FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.