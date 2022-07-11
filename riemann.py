import os
import sys
import math
import time
import heapq
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from data import randomSample
from torch.nn import init
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def trainingdata(low=0, up=0.2, left=-1, right=1, bd_size=100, dim=2):
    # collocation points on boundary
    x_l_bd_list = []
    x_r_bd_list = []
    x_dm_list = []
    # 左边界
    x_bound_l = np.random.uniform(low, up, [bd_size*4, dim])
    x_bound_l[:,0] = left
    
    x_l_bd_list.append(x_bound_l)
    # 左边初值t=0
    x_bound = np.random.uniform(left, (right+left)/2, [bd_size*4, dim])
    x_bound[:,1] = low

    x_l_bd_list.append(x_bound)

    # 右边界
    x_bound_r = np.random.uniform(low, up, [bd_size*4, dim])
    x_bound_r[:,0] = right
    
    x_r_bd_list.append(x_bound_r)
    # 右边初值t=0
    x_bound = np.random.uniform((right+left)/2, right, [bd_size*4, dim])
    x_bound[:,1] = low

    x_r_bd_list.append(x_bound)

    x_l_bd = np.concatenate(x_l_bd_list, axis=0)
    x_r_bd = np.concatenate(x_r_bd_list, axis=0)

    # collocation points inside
    x_dm = np.random.uniform(low, up, [bd_size**2, dim])
    x_dm[:,0] = (x_dm[:,0] - low) * (right - left) / (up - low) + left
    x_dm_list.append(x_dm)

    # x_dm = np.random.uniform(low, up, [3 * bd_size**2, dim])
    # x_dm[:,0] = x_dm[:,0] * (7/2) - 0.3 # -0.3 ~ 0.4
    # x_dm_list.append(x_dm)

    x_dm = np.concatenate(x_dm_list, axis=0)

    return x_l_bd, x_r_bd, x_dm


def drawsample(x, y, x_bd, y_bd, title):
    plt.clf()
    plt.scatter(x, y, marker='.', c='b', s=0.01)
    plt.scatter(x_bd, y_bd, marker='.', c='r', s=0.01)
    plt.savefig('./test_r/' + title + '.png', dpi = 1600)
    plt.clf()


class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
        return 
    
    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class relu2(nn.Module):
    def __init__(self):
        super(relu2, self).__init__()
        return
        
    def forward(self, x):
        x = F.relu(x) * F.relu(x)
        return x


class MhdNet(nn.Module):
    def __init__(self, layers, ul, ur):
        super(MhdNet, self).__init__()
        self.layers = layers
        self.ul = ul
        self.ur = ur

        'activation function'
        self.activation = nn.Tanh()
        # self.activation = nn.ReLU()
        # self.activation = Swish()

        'loss function'
        self.loss_function = nn.MSELoss(reduction='mean')

        'Initialise neural network as a list using nn.Modulelist'
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        'Xavier Normal Initialization'
        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers) - 1):
            # weights from a normal distribution with
            # Recommended gain value for tanh = 5/3?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)

            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)


    def forward(self, x):
        x.requires_grad_(True)

        for i in range(len(self.layers) - 2):
            z = self.linears[i](x)

            x = self.activation(z)

        u = self.linears[-1](x)

        return u


    def loss_IC(self, x, y):

        loss = self.loss_function(self.forward(x), y)

        loss = torch.sqrt(loss)
        return loss


    def loss_BC(self, x, y):

        loss = self.loss_function(self.forward(x), y)

        loss = torch.sqrt(loss)
        return loss


    def loss_PDE(self, x):

        x_1_f = x[:, [0]]
        x_2_f = x[:, [1]]

        g = x.clone()
        g.requires_grad = True
        u = self.forward(g)

        V1 = u[:, [0]] # ρ
        V2 = u[:, [1]] # u
        V3 = u[:, [2]] # p

        U1 = V1 # ρ
        U2 = V1 * V2 # ρu
        U3 = V3 / (5/3 - 1) + (V1 / 2) * (V2**2) # E
        U = torch.cat((U1,U2,U3), 1)

        F1 = V1 * V2  # ρu
        F2 = V1 * (V2**2) + V3 # ρu^2+p
        F3 = V2 * (V3 +  V3 / (5/3 - 1) + (V1 / 2) * (V2**2)) # u(p+E)
        F = torch.cat((F1,F2,F3), 1)
        
        loss = 0
        loss_list = torch.zeros(x.shape[0], 1).to(device)
        for i in range(U.shape[1]):
            u = U[:,i]
            f = F[:,i]
            # U对t求导
            u_x = \
            autograd.grad(u, g, torch.ones_like(u).to(device), retain_graph=True, create_graph=True)[0]

            u_x_2 = u_x[:, 1:2]

            # F对x求导
            f_x = \
            autograd.grad(f, g, torch.ones_like(f).to(device), retain_graph=True, create_graph=True)[0]

            f_x_1 = f_x[:, 0:1]

            # F对t求导
            # f_x_2 = f_x[:, 1:2]

            f = u_x_2 + f_x_1

            loss_list = loss_list + f**2
            loss += self.loss_function(f, torch.zeros(f.shape[0], 1).to(device))

        # loss = torch.sqrt(loss)
        return loss, loss_list


class PartialEq():
    def __init__(self, ul, ur, model_path=''):
        self.layers = np.array([2, 128, 64, 64, 3])
        self.ul = ul
        self.ur = ur
        if model_path:
            self.model = torch.load(model_path)
        else:
            self.model = MhdNet(self.layers, ul, ur)
        self.model.to(device)
        self.lr = 1e-3
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        # 随机采样配合L-BFGS
        self.optimizer2 = torch.optim.LBFGS(self.model.parameters(), lr=self.lr, 
                                            max_iter = 500, 
                                            max_eval = None, 
                                            # tolerance_grad = sys.float_info.min, 
                                            # tolerance_change = 1e-09, 
                                            history_size = 50, 
                                            line_search_fn = 'strong_wolfe')

        self.rate = 0.1
        self.alpha = 1
        self.beta = 1

        self.time_now = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
        logging.basicConfig(filename=os.path.join('./logs',self.time_now + '.log'),
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG)


    def mkdir(self, path):
        path=path.strip()
        path=path.rstrip("\\")

        if os.path.exists(path):
            return False
        else:
            os.makedirs(path) 
            return True


    def train(self, epoch, trainloader_l, trainloader_r, trainloader_f, logs=None, rar=False):

        print('\nEpoch: %d' % epoch)
        if logs:
            logging.info('Epoch: %d' % epoch)

        self.model.train()
        train_loss = 0

        for batch_idx, data in enumerate(zip(trainloader_l, trainloader_r, trainloader_f)):

            x_l_train = data[0]
            x_r_train = data[1]
            x_f_train = data[2]

            l_train = torch.zeros(x_l_train.shape[0], 3).to(device)
            r_train = torch.zeros(x_r_train.shape[0], 3).to(device)
            for i in range(l_train.shape[1]):
                l_train[:,i] = self.ul[i]
                r_train[:,i] = self.ur[i]

            self.optimizer.zero_grad()  # 梯度先全部降为0

            PDE_loss, loss_list = self.model.loss_PDE(x_f_train)
            IC_loss = self.model.loss_IC(x_l_train, l_train)
            BC_loss = self.model.loss_BC(x_r_train, r_train)

            self.optimizer.zero_grad()  # 梯度全部降为0
            # loss.backward()  # 反向传递过程

            # 计算权重
            # PDE_loss.backward(retain_graph=True)
            # grad_res = []
            # for i in range(self.layers):
            #     grad_res.append(torch.max(torch.abs(self.weights[i].grad)))
            # max_grad_res = torch.max(torch.stack(grad_res))
            # self.optimizer.zero_grad()

            # if IC_loss != 0 :
            #     IC_loss.backward(retain_graph=True)
            #     grad_ics = []
            #     for i in range(self.layers-1):
            #         grad_ics.append(torch.mean(torch.abs(self.weights[i].grad)))
            #     mean_grad_ics = torch.mean(torch.stack(grad_ics))
            #     self.optimizer.zero_grad()

            #     adaptive_constant_ics = max_grad_res / mean_grad_ics
            #     self.alpha = self.alpha * (1.0 - self.rate) + self.rate * adaptive_constant_ics
            
            # if BC_loss != 0 :
            #     BC_loss.backward(retain_graph=True)
            #     grad_bcs = []
            #     for i in range(self.layers-1):
            #         grad_bcs.append(torch.mean(torch.abs(self.weights[i].grad)))
            #     mean_grad_bcs = torch.mean(torch.stack(grad_bcs))
            #     self.optimizer.zero_grad()
    
            #     adaptive_constant_bcs = max_grad_res / mean_grad_bcs            
            #     self.beta = self.beta * (1.0 - self.rate) + self.rate * adaptive_constant_bcs

            loss = PDE_loss + self.alpha * IC_loss + self.beta * BC_loss
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            self.optimizer.zero_grad()

            # RAR
            # point_list = []
            # top_N = len(inputs) // 10
            # index_list = map(PDE_loss_list.index, heapq.nlargest(top_N, PDE_loss_list)) #求最大的N个索引
            # for i in list(index_list):
            #     point_list.append(inputs[i])

            # if PDE_loss < 2.0:
            #     optimizer.step()  # 以学习效率lr来优化梯度
            #     train_loss += loss.item()
            # else:
            #     optimizer.zero_grad()  # 梯度全部降为0

            print('PDE loss: %.10f  |  IC loss: %.10f  |  BC loss: %.10f' % (PDE_loss, IC_loss, BC_loss))
            if logs:
                logging.info('PDE loss: %.10f  |  IC loss: %.10f  |  BC loss: %.10f  |  α:%.3f, β:%.3f' \
                % (PDE_loss, IC_loss, BC_loss, self.alpha, self.beta))
            
        print('Train loss: %.10f' % (train_loss / (batch_idx + 1)))
        if logs:
            logging.info('Train loss: %.10f' % (train_loss / (batch_idx + 1)))
        return train_loss / (batch_idx + 1)


    def test(self, epoch, trainloader_l, trainloader_r, trainloader_f, logs=None, rar=False):

        self.model.eval()
        test_loss = 0

        if rar:
            X_l_train_np_array, X_r_train_np_array, X_f_train_np_array = trainingdata(0,2,-2.5,2.5,200)

            X_f_train = torch.from_numpy(X_f_train_np_array).float().to(device)
            X_l_train = torch.from_numpy(X_l_train_np_array).float().to(device)
            X_r_train = torch.from_numpy(X_r_train_np_array).float().to(device)

            trainloader_l = DataLoader(X_l_train, batch_size=100000, num_workers=0, shuffle=True)
            trainloader_r = DataLoader(X_r_train, batch_size=100000, num_workers=0, shuffle=True)
            trainloader_f = DataLoader(X_f_train, batch_size=400000, num_workers=0, shuffle=True)

        for batch_idx, data in enumerate(zip(trainloader_l, trainloader_r, trainloader_f)):

            x_l_train = data[0]
            x_r_train = data[1]
            x_f_train = data[2]

            l_train = torch.zeros(x_l_train.shape[0], 3).to(device)
            r_train = torch.zeros(x_r_train.shape[0], 3).to(device)
            for i in range(l_train.shape[1]):
                l_train[:,i] = self.ul[i]
                r_train[:,i] = self.ur[i]

            self.optimizer.zero_grad()  # 梯度先全部降为0

            PDE_loss, loss_list = self.model.loss_PDE(x_f_train)
            IC_loss = self.model.loss_IC(x_l_train, l_train)
            BC_loss = self.model.loss_BC(x_r_train, r_train)

            self.optimizer.zero_grad()  # 梯度全部降为0
            # loss.backward()  # 反向传递过程

            loss = PDE_loss + self.alpha * IC_loss + self.beta * BC_loss
            # loss.backward()
            # self.optimizer.step()
            test_loss += loss.item()
            self.optimizer.zero_grad()

            # RAR
            loss_list = loss_list.cpu()
            loss_list = loss_list.detach().numpy()
            loss_list = list(loss_list)
            point_list = []
            top_N = len(x_f_train) // 10 # 前10%
            if rar:
                index_list = map(loss_list.index, heapq.nlargest(top_N, loss_list)) #求最大的N个索引
                for i in list(index_list):
                    point_list.append(x_f_train[i])
            else:
                index_list = map(loss_list.index, heapq.nsmallest(top_N, loss_list)) #求最小的N个索引

            print('PDE loss: %.10f  |  IC loss: %.10f  |  BC loss: %.10f' % (PDE_loss, IC_loss, BC_loss))
            if logs:
                logging.debug('PDE loss: %.10f  |  IC loss: %.10f  |  BC loss: %.10f  |  α:%.3f, β:%.3f' \
                % (PDE_loss, IC_loss, BC_loss, self.alpha, self.beta))
            
        print('Test loss: %.10f' % (test_loss / (batch_idx + 1)))
        if logs:
            logging.debug('Test loss: %.10f' % (test_loss / (batch_idx + 1)))
        if rar:
            return point_list
        else:
            return list(index_list)


    def finetune(self, epoch, trainloader_l, trainloader_r, trainloader_f, logs=None):

        print('\nEpoch: %d' % epoch)
        if logs:
            logging.info('Epoch: %d' % epoch)

        self.model.train()
        train_loss = 0

        for batch_idx, data in enumerate(zip(trainloader_l, trainloader_r, trainloader_f)):

            x_l_train = data[0]
            x_r_train = data[1]
            x_f_train = data[2]
                            
            l_train = torch.zeros(x_l_train.shape[0], 3).to(device)
            r_train = torch.zeros(x_r_train.shape[0], 3).to(device)
            for i in range(l_train.shape[1]):
                l_train[:,i] = self.ul[i]
                r_train[:,i] = self.ur[i]

            def closure():
                self.optimizer2.zero_grad()  # 梯度先全部降为0

                PDE_loss, _ = self.model.loss_PDE(x_f_train)
                IC_loss = self.model.loss_IC(x_l_train, l_train)
                BC_loss = self.model.loss_BC(x_r_train, r_train)

                if logs:
                    logging.debug('PDE loss: %.10f  |  IC loss: %.10f  |  BC loss: %.10f  |  α:%.3f, β:%.3f' \
                    % (PDE_loss, IC_loss, BC_loss, self.alpha, self.beta))
                self.optimizer2.zero_grad()  # 梯度全部降为0
                loss = PDE_loss + self.alpha * IC_loss + self.beta * BC_loss
                loss.backward()

                return loss

            self.optimizer2.step(closure)

            self.optimizer2.zero_grad()  # 梯度先全部降为0
            PDE_loss, _ = self.model.loss_PDE(x_f_train)
            IC_loss = self.model.loss_IC(x_l_train, l_train)
            BC_loss = self.model.loss_BC(x_r_train, r_train)
            loss = PDE_loss + self.alpha * IC_loss + self.beta * BC_loss
            train_loss += loss.item()
            self.optimizer2.zero_grad()

            print('PDE loss: %.10f  |  IC loss: %.10f  |  BC loss: %.10f' % (PDE_loss, IC_loss, BC_loss))
            if logs:
                logging.info('PDE loss: %.10f  |  IC loss: %.10f  |  BC loss: %.10f  |  α:%.3f, β:%.3f' \
                % (PDE_loss, IC_loss, BC_loss, self.alpha, self.beta))
            
        print('Train loss: %.10f' % (train_loss / (batch_idx + 1)))
        if logs:
            logging.info('Train loss: %.10f' % (train_loss / (batch_idx + 1)))
        return train_loss / (batch_idx + 1)


    def run(self, epochs=10000, logs=True, randomsample=False, rar=False):

        #获取当前时间
        
        model_path = os.path.join('./model', self.time_now)
        self.mkdir(model_path)

        X_l_train_np_array, X_r_train_np_array, X_f_train_np_array = trainingdata(0, 2, -2.5, 2.5, 200)
        X_i = np.concatenate([X_l_train_np_array, X_r_train_np_array])

        drawsample(X_f_train_np_array[:,0], X_f_train_np_array[:,1], X_i[:,0], X_i[:,1], 'sample_points')

        X_f_train = torch.from_numpy(X_f_train_np_array).float().to(device)
        X_l_train = torch.from_numpy(X_l_train_np_array).float().to(device)
        X_r_train = torch.from_numpy(X_r_train_np_array).float().to(device)

        trainloader_l = DataLoader(X_l_train, batch_size=100000, num_workers=0, shuffle=True)
        trainloader_r = DataLoader(X_r_train, batch_size=100000, num_workers=0, shuffle=True)
        trainloader_f = DataLoader(X_f_train, batch_size=400000, num_workers=0, shuffle=True)


        for epoch in range(epochs):
            train_loss = self.train(epoch, trainloader_l, trainloader_r, trainloader_f, logs)
            # 保存模型
            if (epoch + 1) % (epochs/(1000 if epochs >= 1000 else epochs)) == 0: 
                torch.save(self.model, os.path.join(model_path,'model_%003d_TrainLoss=%.10f%%.pkl' \
                % (epoch // (epochs/(1000 if epochs >= 1000 else epochs)), train_loss)))
            if rar and epoch < epochs / 2 and (epoch + 1) % (epochs / 20) == 0: # 自适应采样
                point_list = self.test(epoch, trainloader_l, trainloader_r, trainloader_f, logs, rar=False)
                X_f_train_np_array = np.delete(X_f_train_np_array, point_list, 0) # 删除损失小的点
                point_list = self.test(epoch, trainloader_l, trainloader_r, trainloader_f, logs, rar=True) # list[tensor]
                for i in point_list:
                    point_array = np.array([(i[0].item(), i[1].item())]) # np.array
                    X_f_train_np_array = np.concatenate([X_f_train_np_array, point_array]) # 拼接，加入损失大的点
                drawsample(X_f_train_np_array[:,0], X_f_train_np_array[:,1], X_i[:,0], X_i[:,1], 'asample_points')
                X_f_train = torch.from_numpy(X_f_train_np_array).float().to(device) # 转 tensor
                trainloader_f = DataLoader(X_f_train, batch_size=400000, num_workers=0, shuffle=True)
            if (epoch + 1) % (epochs/2) == 0:
                self.lr = self.lr / 10
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                for param_group in self.optimizer2.param_groups:
                    param_group['lr'] = self.lr

        # self.lr = 1e-3
        # for param_group in self.optimizer2.param_groups:
        #     param_group['lr'] = self.lr
        for epoch in range(epochs, epochs + epochs//5):

            train_loss = self.finetune(epoch, trainloader_l, trainloader_r, trainloader_f, logs)
            # 保存模型
            if (epoch + 1) % (epochs/(1000 if epochs >= 1000 else epochs)) == 0: 
                torch.save(self.model, os.path.join(model_path,'finetune_model_%003d_TrainLoss=%.10f%%.pkl' \
                % (epoch // (epochs/(1000 if epochs >= 1000 else epochs)), train_loss)))
