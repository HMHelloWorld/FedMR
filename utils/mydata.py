# import os
# import torch
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
#
#
# MNIST_MEAN = (0.1307,)
# MNIST_STD = (0.3081,)
#
# CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
# CIFAR10_STD = (0.2023, 0.1994, 0.2010)
#
# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)
#
# def get_dataset(dset_name, batch_size=128, n_worker=4, data_root='../../dataset'):
#
#     print('=> Preparing data..')
#     kwargs = {'num_workers': n_worker, 'pin_memory': True} if torch.cuda.is_available() else {}
#
#     if dset_name == 'mnist':
#         #normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
#         transform_train = transforms.Compose([
#             transforms.ToTensor(),
#             #normalize,
#         ])
#         trainset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform_train)
#         train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, **kwargs)
#
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             #normalize,
#         ])
#         testset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform_test)
#         test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, **kwargs)
#
#         n_class = 10
#
#     elif dset_name == 'cifar10':
#
#         #normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             #normalize,
#         ])
#         trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
#         train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, **kwargs)
#
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             #normalize,
#         ])
#         testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
#         test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, **kwargs)
#
#         n_class = 10
#
#     elif dset_name == 'imagenet':
#
#         traindir = os.path.join(data_root, 'imagenet/train')
#         valdir = os.path.join(data_root, 'imagenet/val')
#
#         #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         transform_train = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             #normalize,
#         ])
#         trainset = datasets.ImageFolder(traindir, transform_train)
#         train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, **kwargs)
#
#         transform_test = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             #normalize,
#         ])
#         testset = datasets.ImageFolder(valdir, transform_test)
#         test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, **kwargs)
#
#         n_class = 1000
#
#     else:
#         raise NotImplementedError
#
#     return train_loader, test_loader, n_class
#
#
# def get_testloader(dset_name, batch_size=128, n_worker=4, data_root='../../dataset', subset_idx=None):
#     print('=> Preparing testing data..')
#     kwargs = {'num_workers': n_worker, 'pin_memory': True} if torch.cuda.is_available() else {}
#     if dset_name == 'mnist':
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#         ])
#         testset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform_test)
#         if subset_idx is not None:
#             testset = torch.utils.data.Subset(testset, subset_idx)
#         test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, **kwargs)
#         n_class = 10
#     elif dset_name == 'cifar10':
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#         ])
#         testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
#         if subset_idx is not None:
#             testset = torch.utils.data.Subset(testset, subset_idx)
#         test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, **kwargs)
#         n_class = 10
#     elif dset_name == 'imagenet':
#         valdir = os.path.join(data_root, 'imagenet/val')
#         transform_test = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#         ])
#         testset = datasets.ImageFolder(valdir, transform_test)
#         if subset_idx is not None:
#             testset = torch.utils.data.Subset(testset, subset_idx)
#         test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, **kwargs)
#         n_class = 1000
#     else:
#         raise NotImplementedError
#     return test_loader, n_class
from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.cifar import CIFAR10

class CIFAR100_coarse(CIFAR10):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'coarse_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['coarse_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()