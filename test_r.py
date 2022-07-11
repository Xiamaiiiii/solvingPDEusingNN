from riemann import *
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
    plt.savefig('./test_r/Loss.png')


def test_one(model_path, test_path, result_path):
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = torch.load(model_path)
    net = net.to(device)

    VL = torch.tensor([1,       0,  1.0])
    VR = torch.tensor([0.125,   0,  0.1])
    # VL = torch.tensor([1.4,     0.1,    1.0])
    # VR = torch.tensor([1.0,     0.1,    1.0])
    VL = VL.to(device)
    VR = VR.to(device)
   
    test = np.loadtxt(test_path, delimiter=',')
    test = torch.from_numpy(test).float()
    test_x, test_y = test.split(2,dim=1)
    testset = TensorDataset(test_x, test_y)

    testloader = DataLoader(testset, batch_size=100000, shuffle=False)
    
    z = (np.zeros(40401)).reshape(201, 201)    
    z1 = (np.zeros(40401)).reshape(201, 201)    
    z2 = (np.zeros(40401)).reshape(201, 201)
    z3 = (np.zeros(40401)).reshape(201, 201)

    net.eval()
    result = []
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        # print(inputs)
        loss, loss_list = net.loss_PDE(inputs)
        outputs = net(inputs)

        inputs = inputs.cpu()
        inputs = inputs.detach().numpy()
        outputs = outputs.cpu()
        outputs = outputs.detach().numpy()
        loss_list = loss_list.cpu()
        loss_list = loss_list.detach().numpy()
        
        outputs = np.append(outputs, loss_list, axis=1)

        for y in range(201):
            for x in range(201):
                z[x][y] = outputs[201*y+x, 0]
                z1[x][y] = outputs[201*y+x, 1]
                z2[x][y] = outputs[201*y+x, 2]
                z3[x][y] = outputs[201*y+x, 3]
        
        outputs = np.append(inputs, outputs, axis=1)
        # result.append(outputs)

    # result.sort(key=lambda x:x[0])
    # result = np.array(result)
    # data = result
    data = outputs
    
    np.savetxt(result_path, data, fmt='%.8f', delimiter=',')
    
    X = data[:,0]
    Y1 = data[:,2]
    Y2 = data[:,3]
    Y3 = data[:,4]
    loss = data[:,5]

    # 画图
    figure = ['Figure1','Figure2','Figure3','Figure4']
    Y = [Y1,Y2,Y3,loss]
    Ylim = [(0.0,1.5),(-0.1,1.1),(0.0,1.1),(-0.01,0.1)]
    Ylabel = ['ρ','u','p','PDE_Loss']

    # for i in range(len(Y)):
    #     fig = plt.figure(figure[i],figsize=(10,10)).add_subplot(111)
    #     fig.plot(X, Y[i], c='g', marker='.', linestyle='--')
    #     fig.set_title(' ')
    #     fig.set_xlabel('x')
    #     fig.set_ylabel(Ylabel[i])
    #     fig.set_xlim(-2.5,2.5)
    #     fig.set_ylim(Ylim[i][0],Ylim[i][1])
    #     fig.ticklabel_format(useOffset=False, style='plain')
    #     plt.savefig('./test_r/' + Ylabel[i] + '.png')
 
    plt.clf()
    plt.imshow(z, extent=(0, 1, 0, 1), cmap=plt.cm.rainbow, origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig('./test_r/' + 'hot_rho' + '.png', dpi = 800) 
    plt.clf()
    plt.imshow(z1, extent=(0, 1, 0, 1), cmap=plt.cm.rainbow, origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig('./test_r/' + 'hot_u' + '.png', dpi = 800) 
    plt.clf()
    plt.imshow(z2, extent=(0, 1, 0, 1), cmap=plt.cm.rainbow, origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig('./test_r/' + 'hot_p' + '.png', dpi = 800)
    plt.clf()
    plt.imshow(z3, extent=(0, 1, 0, 1), cmap=plt.cm.rainbow, origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig('./test_r/' + 'hot_loss' + '.png', dpi = 800)


if __name__ == '__main__':

    draw_loss('./logs/2022-06-20-16:55.log')
    test_one(model_path='./model/2022-06-20-16:55/finetune_model_1000_TrainLoss=0.0273135286%.pkl', \
    test_path='./train25.csv', result_path='./test_r/result.csv')