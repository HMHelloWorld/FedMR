import matplotlib
matplotlib.use('Agg')
import copy

from utils.options import args_parser
from utils.set_seed import set_random_seed
from models.Update import *
from models.Nets import *
from models.MobileNetV2 import MobileNetV2
from models.Fed import Aggregation, Weighted_Aggregation_FedASync
from models.test import test_img
from models.resnetcifar import *
from models import *
from utils.get_dataset import get_dataset
from utils.utils import save_result,save_model
from Algorithm.Training_FedGen import FedGen
from Algorithm.Triaining_Scaffold import Scaffold
from Algorithm.Training_FedDC import FedDC
from Algorithm.Training_FedCross import FedCross
from Algorithm.Training_FedMR import FedMR
from Algorithm.Training_FedMR import FedMR_Frozen
from Algorithm.Training_FedMR import FedMR_Partition
from Algorithm.Training_CFL import CFL
from Algorithm.Training_FedIndenp import FedIndep
from Algorithm.Training_Asyn_FedSA import FedSA
from Algorithm.Training_Asyn_GitFL import GitFL
from utils.Clients import Clients
import utils.asynchronous_client_config as AsynConfig


if __name__ == '__main__':
    
    PATH = "/home/huming/hm/FederatedLearning/loss-landscape-master/model/cifar10_FedMR_resnet20_test_model_1000_lr_0.01_2023_05_13_14_23_43_frac_0.1.txt"
    model_dict=torch.load(PATH)
    model_dict=model.load_state_dict(torch.load(PATH))