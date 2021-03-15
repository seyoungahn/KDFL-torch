"""
CIFAR-10 data normalization reference:
https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""

import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def fetch_dataloader(types, params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    :param types:
    :param params:
    :return:
    """
    # Using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    else:
        # Data augmentation can be turned off
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    # Transformer for dev set
    test_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    data_path = params.dataset_path + '/' + params.dataset_type
    if params.dataset_type == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transformer)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)
    testloader = torch.utils.data.DataLoader(testset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dataloader = trainloader
    else:
        dataloader = testloader

    return dataloader

def fetch_subset_dataloader(types, params):
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    :param types:
    :param params:
    :return:
    """
    # Using random crops and horizontal flip for trainset
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    else:
        # Data augmentation can be turned off
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    # Transformer for testset
    test_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    data_path = params.dataset_path + '/' + params.dataset_type
    if params.dataset_type == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transformer)

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)
    testloader = torch.utils.data.DataLoader(testset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dataloader = trainloader
    else:
        dataloader = testloader

    return dataloader