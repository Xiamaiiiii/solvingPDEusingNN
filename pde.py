from riemann import *
import os
import time
from data import *
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def data_loader(data_path):

    train = np.loadtxt(data_path, delimiter=',')
    train = torch.from_numpy(train).float()
    
    train_x, train_y = train.split([2,1], dim=1)
    return train_x

    # train_x, train_y = train.split([2,7], dim=1)
    # trainset = TensorDataset(train_x, train_y)
    # trainloader = DataLoader(trainset, batch_size=600, num_workers=0, shuffle=True)

    # return trainloader


if __name__ == '__main__':
    # start training and testing
    # The first problem is one of the familiar shock tube problem presented by Dai & Woodward (1994).
    VL = torch.tensor([1,       0,      1.0])
    VR = torch.tensor([0.125,   0,      0.1])

    # VL = torch.tensor([1.4,     0.1,    1.0])
    # VR = torch.tensor([1.0,     0.1,    1.0])

    # trainset = data_loader('train01.csv')

    # model = PartialEq(VL, VR, BX, model_path='./model/2021-10-14-17:27/model_626_TrainLoss=0.0369696054%.pkl') # mhd128
    # model = PartialEq(VL, VR, BX, model_path='./model/2021-10-29-17:11/model_316_TrainLoss=0.0219123636%.pkl') # mhd648
    model = PartialEq(VL, VR)

    model.run(epochs=50000, randomsample=False, rar=True)

    # trainloader = data_loader('gt.csv')

    # model.run(trainloader, epochs=1000, randomsample=False, rar=False)
