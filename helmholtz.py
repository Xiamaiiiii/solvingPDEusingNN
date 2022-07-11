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

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def q(x, y):
    a1 = 1
    a2 = 4
    k = 0
    return (- (a1 * np.pi) ** 2 - (a2 * np.pi) ** 2 + k ** 2) * torch.sin(a1 * np.pi * x) * torch.sin(a2 * np.pi * y)


def trainingdata(low=0, up=1, bd_size=100, dim=2):
    # collocation points on boundary
    x_bd_list = []
    for i in range(dim):
        x_bound = np.random.uniform(low, up, [bd_size, dim])
        x_bound[:,i] = low
        x_bd_list.append(x_bound)
        x_bound = np.random.uniform(low, up, [bd_size, dim])
        x_bound[:,i] = up
        x_bd_list.append(x_bound)
    x_bd = np.concatenate(x_bd_list, axis=0)

    # collocation points inside
    x_dm = np.random.uniform(low, up, [bd_size**2, dim])

    return x_bd, x_dm


class MhdNet(nn.Module):
    def __init__(self, layers):
        super(MhdNet, self).__init__()
        self.layers = layers

        'activation function'
        self.activation = nn.Tanh()

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

    def loss_BC(self, x, y):

        loss_u = self.loss_function(self.forward(x), y)

        return loss_u

    def loss_PDE(self, x):

        x_1_f = x[:, [0]]
        x_2_f = x[:, [1]]

        g = x.clone()

        g.requires_grad = True

        u = self.forward(g)

        u_x = \
        autograd.grad(u, g, torch.ones([x.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]

        u_x_1 = u_x[:, 0:1]
        u_x_2 = u_x[:, 1:2]

        u_xx_1 = autograd.grad(u_x_1, g, torch.ones(u_x_1.shape).to(device), create_graph=True)[0]
        u_xx_2 = autograd.grad(u_x_2, g, torch.ones(u_x_2.shape).to(device), create_graph=True)[0]

        u_xx_11 = u_xx_1[:, 0:1]
        u_xx_22 = u_xx_2[:, 1:2]

        f = u_xx_11 + u_xx_22 + u - q(x_1_f, x_2_f)

        loss_f = self.loss_function(f, torch.zeros(f.shape[0], 1).to(device))

        return loss_f


class PartialEq():
    def __init__(self, ul, ur, Bx, model_path=''):
        self.layers = np.array([2, 50, 50, 50, 50, 50, 1])
        if model_path:
            self.model = torch.load(model_path)
        else:
            self.model = MhdNet(self.layers)
        self.model.to(device)
        self.lr = 1e-3
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        # 随机采样配合L-BFGS
        self.optimizer2 = torch.optim.LBFGS(self.model.parameters(), lr=self.lr, 
                                            max_iter = 50, 
                                            max_eval = None, 
                                            tolerance_grad = sys.float_info.min, 
                                            tolerance_change = sys.float_info.min, 
                                            history_size = 100, 
                                            line_search_fn = 'strong_wolfe')

        self.rate = 0.1
        self.alpha = 1
        self.beta = 1

        self.time_now = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
        logging.basicConfig(filename=os.path.join('./logs',self.time_now + '.log'),
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG)


    def initNetParams(self):
        '''Init net parameters.'''
        weights = []
        biases = []
        layers = 0
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=1)
                weights.append(m.weight)
                biases.append(m.bias)
                layers += 1
        return weights, biases, layers


    def mkdir(self, path):
        path=path.strip()
        path=path.rstrip("\\")

        if os.path.exists(path):
            return False
        else:
            os.makedirs(path) 
            return True


    def train(self, epoch, trainloader_u, trainloader_f, logs=None):

        print('\nEpoch: %d' % epoch)
        if logs:
            logging.info('Epoch: %d' % epoch)

        self.model.train()
        train_loss = 0

        for batch_idx, data in enumerate(zip(trainloader_u, trainloader_f)):

            x_u_train = data[0]
            x_f_train = data[1]
            
            self.optimizer.zero_grad()  # 梯度先全部降为0

            PDE_loss = self.model.loss_PDE(x_f_train)
            BC_loss = self.model.loss_BC(x_u_train, torch.zeros(x_u_train.shape[0], 1).to(device))

            self.optimizer.zero_grad()  # 梯度全部降为0
            # loss.backward()  # 反向传递过程

            loss = PDE_loss + self.alpha * BC_loss
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            self.optimizer.zero_grad()

            # if PDE_loss < 2.0:
            #     optimizer.step()  # 以学习效率lr来优化梯度
            #     train_loss += loss.item()
            # else:
            #     optimizer.zero_grad()  # 梯度全部降为0

            print('PDE loss: %.10f  |  IC loss: %.10f' % (PDE_loss, BC_loss))
            if logs:
                logging.info('PDE loss: %.10f  |  IC loss: %.10f  |  α:%.3f, β:%.3f' \
                % (PDE_loss, BC_loss, self.alpha, self.beta))
            
        print('Train loss: %.10f' % (train_loss / (batch_idx + 1)))
        if logs:
            logging.info('Train loss: %.10f' % (train_loss / (batch_idx + 1)))
        return train_loss / (batch_idx + 1)


    def finetune(self, epoch, trainloader_u, trainloader_f, logs=None):

        print('\nEpoch: %d' % epoch)
        if logs:
            logging.info('Epoch: %d' % epoch)

        self.model.train()
        train_loss = 0

        for batch_idx, data in enumerate(zip(trainloader_u, trainloader_f)):

            x_u_train = data[0]
            x_f_train = data[1]        

            def closure():
                self.optimizer2.zero_grad()  # 梯度先全部降为0

                PDE_loss = self.model.loss_PDE(x_f_train)
                BC_loss = self.model.loss_BC(x_u_train, torch.zeros(x_u_train.shape[0], 1).to(device))

                if logs:
                    logging.debug('PDE loss: %.10f  |  BC loss: %.10f  |  α:%.3f, β:%.3f' \
                    % (PDE_loss, BC_loss, self.alpha, self.beta))
                self.optimizer2.zero_grad()  # 梯度全部降为0
                loss = PDE_loss + self.alpha * BC_loss
                loss.backward()

                return loss

            self.optimizer2.step(closure)

            self.optimizer2.zero_grad()  # 梯度先全部降为0
            PDE_loss = self.model.loss_PDE(x_f_train)
            BC_loss = self.model.loss_BC(x_u_train, torch.zeros(x_u_train.shape[0], 1).to(device))
            loss = PDE_loss + self.alpha * BC_loss
            train_loss += loss.item()
            self.optimizer2.zero_grad()

            print('PDE loss: %.10f  |  BC loss: %.10f' % (PDE_loss, BC_loss))
            if logs:
                logging.info('PDE loss: %.10f  |  BC loss: %.10f  |  α:%.3f, β:%.3f' \
                % (PDE_loss, BC_loss, self.alpha, self.beta))
            
        print('Train loss: %.10f' % (train_loss / (batch_idx + 1)))
        if logs:
            logging.info('Train loss: %.10f' % (train_loss / (batch_idx + 1)))
        return train_loss / (batch_idx + 1)


    def run(self, epochs=10000, logs=True, randomsample=False, rar=False):

        #获取当前时间
        model_path = os.path.join('./model', self.time_now)
        self.mkdir(model_path)
  
        X_u_train_np_array, X_f_train_np_array = trainingdata()
        X_f_train = torch.from_numpy(X_f_train_np_array).float().to(device)
        X_u_train = torch.from_numpy(X_u_train_np_array).float().to(device)

        trainloader_u = DataLoader(X_u_train, batch_size=400, num_workers=0, shuffle=True)
        trainloader_f = DataLoader(X_f_train, batch_size=10000, num_workers=0, shuffle=True)


        for epoch in range(epochs):

            train_loss = self.train(epoch, trainloader_u, trainloader_f, logs)
            # 保存模型
            if (epoch + 1) % (epochs/(1000 if epochs >= 1000 else epochs)) == 0: 
                torch.save(self.model, os.path.join(model_path,'model_%003d_TrainLoss=%.10f%%.pkl' \
                % (epoch // (epochs/(1000 if epochs >= 1000 else epochs)), train_loss)))
            # if rar and (epoch + 1) % (epochs/4) == 0:
            #     extra_point = self.aroundSampling(point_list)
            #     trainloader = DataLoader(torch.cat((trainset,extra_point)), batch_size=256, shuffle=True)
            if (epoch + 1) % (epochs/4) == 0:
                self.lr = self.lr / 10
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                for param_group in self.optimizer2.param_groups:
                    param_group['lr'] = self.lr

        # self.lr = 1e-3
        # for param_group in self.optimizer2.param_groups:
        #     param_group['lr'] = self.lr
        for epoch in range(epochs, epochs + epochs//5):

            train_loss = self.finetune(epoch, trainloader_u, trainloader_f, logs)
            # 保存模型
            if (epoch + 1) % (epochs/(1000 if epochs >= 1000 else epochs)) == 0: 
                torch.save(self.model, os.path.join(model_path,'finetune_model_%003d_TrainLoss=%.10f%%.pkl' \
                % (epoch // (epochs/(1000 if epochs >= 1000 else epochs)), train_loss)))

            if (epoch + 1) % (epochs/4) == 0:
                self.lr = self.lr / 10
                for param_group in self.optimizer2.param_groups:
                    param_group['lr'] = self.lr
