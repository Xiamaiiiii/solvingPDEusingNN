from helmholtz import *
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

Epoch = 'Epoch'
Loss = 'Train loss'

def draw_loss(log_path):

    x = []
    y = []

    with open(log_path,'r') as f:
        for data in f.readlines()[0:]:
            if 'INFO - ' in data:
                data = data.split('INFO - ')[1]
            else:
                continue
            if data.split(':')[0] == Epoch:
                x.append(int(data.split(':')[-1]))
            if data.split(':')[0] == Loss:
                y.append(float(data.split(':')[-1]))

    if len(x) != len(y):
        x = x[:-1]

    fig=plt.figure(figsize=(10,10)).add_subplot(111)
    fig.plot(x, y, c='g', linewidth=1)
    fig.set_xlabel(Epoch, fontsize=20)
    fig.set_ylabel(Loss, fontsize=20)
    fig.set_xlim(0,max(x))
    fig.set_ylim(min(y),max(y))
    fig.set_yscale('log')
    plt.savefig('./h_test/Loss.png')


def test_one(model_path, test_path, result_path):
    
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    net = torch.load(model_path)
    net = net.to(device)
   
    test = np.loadtxt(test_path, delimiter=',')
    test = torch.from_numpy(test).float()
    test_x, test_y = test.split(2,dim=1)
    testset = TensorDataset(test_x, test_y)

    testloader = DataLoader(testset, batch_size=1, shuffle=None)

    z = (np.zeros(40401)).reshape(201, 201)
    
    net.eval()
    result = []
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        # print(inputs)
        outputs = net(inputs)

        # PDE_loss = net.loss_PDE(inputs)

        inputs = inputs.cpu()
        inputs = inputs.detach().numpy()[0]
        outputs = outputs.cpu()
        outputs = outputs.detach().numpy()[0]
    
        y = int(round(round(inputs[0], 3)/0.005))
        x = int(round(round(inputs[1], 3)/0.005))
        
        z[x][y] = outputs

    np.savetxt(result_path, z, fmt='%.8f', delimiter=',')

    # 画图
    plt.clf()
    plt.imshow(z, extent=(0, 1, 0, 1), cmap=plt.cm.rainbow, origin='lower')
    plt.colorbar()
    plt.savefig('./h_test/' + 'result' + '.png', dpi = 800)


if __name__ == '__main__':

    draw_loss('./logs/2022-05-13-22:34.log')
    test_one(model_path='./model/2022-05-13-22:34/finetune_model_1000_TrainLoss=0.0022961695%.pkl', test_path='./train01.csv', result_path='./h_test/result.csv')