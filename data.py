# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def randomSample(xl,xr,t,size=2**6):

    xe = xr - xl
    te = t

    x = torch.cat((torch.rand([size, 1]) * xe + xl, torch.rand([size, 1]) * te), dim=1)
    x_initial = torch.cat((torch.rand(size//2, 1) * xe, torch.zeros([size//2, 1])), dim=1)
    x_boundary_left = torch.cat((torch.zeros([size//4, 1]) + xl, torch.rand([size//4, 1]) * te), dim=1)
    x_boundary_right = torch.cat((torch.zeros([size//4, 1]) + xr, torch.rand([size//4, 1]) * te), dim=1)

    train_x = torch.cat((x, x_initial, x_boundary_left, x_boundary_right))

    trainloader = DataLoader(train_x, batch_size=size*2, shuffle=True)

    return trainloader


def trainSample():

    data = []

    for x in range(0,201,1):
        for t in range(0,201,1):
            data.append([x*2*np.pi/200.0,t*2*np.pi/200.0,3])

    data = np.array(data)
            
    np.savetxt('train_2d.csv', data, fmt='%.4f', delimiter=',')


def testSample():

    data = []
    t = 2

    # -0.5 ~ 0.5 之间每隔0.005取一个点
    for x in range(-4000,4001,40):
        data.append([x/1000.0,t,0]) 

    data = np.array(data)
            
    np.savetxt('test4.csv', data, fmt='%.3f', delimiter=',')


def realData():
    ground_truth = np.loadtxt('./result.dat')
    
    data = []
    
    np.random.normal(0,0.05,500)

    for i in range(2, ground_truth.shape[0], 2):
        for x in np.random.normal(ground_truth[i][0], 0.05, 500):
            data.append([x, 0.2])

    for i in range(len(data) - 1, -1, -1):
        x = data[i][0]
        for j in range(0, ground_truth.shape[0], 2):
            if x >= ground_truth[j][0] and x < ground_truth[j+1][0]:
                data[i] = data[i] + ground_truth[j].tolist()[1:]
                break
        if len(data[i]) == 2:
            data.remove(data[i])
        

    # # -0.5 ~ 0.5 之间每隔0.01取一个点
    # for x in range(int(ground_truth[0][0]*1000), int(ground_truth[-1][0]*1000) + 1, 1):
    #     for i in range(0, ground_truth.shape[0], 2):
    #         if x/1000.0 >= ground_truth[i][0] and x/1000.0 <ground_truth[i+1][0]:
    #             data.append([x/1000.0, 0.2] + ground_truth[i].tolist()[1:])
    #             break

    data = np.array(data)
            
    np.savetxt('gt.csv', data, fmt='%.5f', delimiter=',')


if __name__ == '__main__':

    trainSample()
    testSample()
    # x = randomSample(-0.5,0.5,0.2)
    # print(x)
    # realData()