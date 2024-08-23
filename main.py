import os
import torch
from copy import deepcopy
import numpy as np
import xarray as xr
import pandas as pd
import torch.nn as nn
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import zipfile
import torchvision.models as models
from openstl.models.mfwpn import MFWPN_Model
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
import pickle
import math
from matplotlib.pyplot import MultipleLocator
from utils.data_sliding import *
import pywt
import pywt.data
import torch.nn.functional as F
from utils import SSIM
from thop import profile

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):

        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        torch.manual_seed(35)
        self.network = MFWPN_Model().to(configs.device)
        adam = torch.optim.Adam([{'params': self.network.parameters()}], lr=0, weight_decay=configs.weight_decay)
        factor = math.sqrt(configs.d_model*configs.warmup)*0.001
        self.opt = NoamOpt(configs.d_model, factor, warmup=configs.warmup, optimizer=adam)
        self.u, self.v = 'u', 'v'

    def loss(self, y_pred, y_true, idx):
        if idx == 'u':
            idx = 0
        if idx == 'v':
            idx = 1
            
        rmse = torch.mean((y_pred[:, :, idx] - y_true[:, :, idx])**2, dim=[2, 3])
        rmse = torch.mean(torch.sqrt(rmse.mean(dim=0)))
            
        return rmse
    
    def SSIM_loss(self, pred, true):
        pred_np = pred[:,:,:2].permute(1,0,2,3,4)
        true_np = true[:,:,:2].permute(1,0,2,3,4)
        total_loss = 0.0
 
        for i in range(pred_np.shape[0]):
                  loss = 1 - SSIM.SSIM(pred_np[i], true_np[i])
                  total_loss += loss.item()

        average_loss = total_loss / (pred_np.shape[0])
        return average_loss

    def Angle_loss(self, batch_y, pred_y):
        true = self.Angle_wind(batch_y)
        pred = self.Angle_wind(pred_y)
        
        min_all = min(true.min(), pred.min())
        max_all = max(pred.max(), true.max())
        true = (true - min_all) / (max_all - min_all)
        pred = (pred - min_all) / (max_all - min_all)
        
        mse = torch.mean((true - pred)**2)
        rmse = mse.sqrt()
        return rmse
    
    def Angle_wind(self, batch_y):
        true = torch.tensor(batch_y, dtype=torch.float)
        a_fushu = true[:, :, 0, :, :]
        b_fushu = true[:, :, 1, :, :]
        complex_tensor = a_fushu + 1j * b_fushu
        angle_rad = torch.angle(complex_tensor)
        angle_deg = angle_rad * (180 / 3.141592653589793)
        angle_metric = angle_deg.unsqueeze(2)
        return angle_metric
    
    def train_once(self, input_uv, uv_true, ssr_ratio, ele):
        uv_pred = self.network(input_uv.float().to(self.device), ele.float().to(self.device))
        self.opt.optimizer.zero_grad()
        loss_u = self.loss(uv_pred, uv_true.float().to(self.device), self.u)
        loss_v = self.loss(uv_pred, uv_true.float().to(self.device), self.v)        

        loss_ssim = self.SSIM_loss(uv_pred, uv_true.float().to(self.device))   
        loss_angle = self.Angle_loss(uv_true.float().to(self.device), uv_pred)

        loss = loss_u + loss_v + loss_ssim + 3.5*loss_angle
 
        loss.backward()
        if configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), configs.clipping_threshold)
        self.opt.step()
        
        return loss_u.item(), loss_v.item(), loss_ssim, loss_angle, loss

    def test(self, dataloader_test, ele):
        uv_pred = []
        with torch.no_grad():
            for input_uv, uv_true in dataloader_test:    
                uv = self.network(input_uv.float().to(self.device), ele.float().to(self.device))
                uv_pred.append(uv)

        return torch.cat(uv_pred, dim=0)
    
    def infer(self, dataset, dataloader, ele):
        self.network.eval()
        with torch.no_grad():
            uvpred = self.test(dataloader, ele)
            uv_true = torch.from_numpy(dataset.target).float().to(self.device)
            
            loss_u = self.loss(uv_pred, uv_true, self.u).item()
            loss_v = self.loss(uv_pred, uv_true, self.v).item()
  
        return loss_u, loss_v

    def train(self, dataset_train, dataset_eval, elev, chk_path):
        torch.manual_seed(0)
        print('loading train dataloader')
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        print('loading eval dataloader')
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)
        elev = torch.tensor(elev)
          
        count = 0
        best = 100
        ssr_ratio = 1
        for i in range(self.configs.num_epochs):
            print('\nepoch: {0}'.format(i+1))
            # train
            self.network.train()
            for j, (input_uv, uv_true) in enumerate(dataloader_train):
                if ssr_ratio > 0:
                    ssr_ratio = max(ssr_ratio - self.configs.ssr_decay_rate, 0)
                    
                loss_u, loss_v,  loss_ssim, loss_angle, loss = self.train_once(input_uv, uv_true, ssr_ratio, elev)
   
                if (j+1) % self.configs.display_interval == 0:
                    print('batch training loss: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, ssr: {:.5f}, lr: {:.5f}'.format(loss_u, loss_v, loss_ssim, loss_angle, loss, ssr_ratio, self.opt.rate()))
              
                if (i+1 >= 10) and (j+1)%(self.configs.display_interval * 2) == 0:
                    loss_u_eval_0, loss_v_eval_0 = self.infer(dataset=dataset_eval, dataloader=dataloader_eval, ele=elev)
                    loss_eval_0 = loss_u_eval_0 + loss_v_eval_0
                    print('batch eval loss: {:.4f}, {:.4f}, {:.4f}'.format(loss_u_eval_0, loss_v_eval_0, loss_eval_0))
                
                    if loss_eval_0 < best:
                        self.save_model(chk_path)
                        best = loss_eval_0
                        count = 0
                        print('saving model')


            loss_u_eval, loss_v_eval = self.infer(dataset=dataset_eval, dataloader=dataloader_eval, ele=elev)
            loss_eval = loss_u_eval + loss_v_eval
            print('epoch eval loss: {:.4f}, {:.4f}, {:.4f}'.format(loss_u_eval, loss_v_eval, loss_eval))
            
            
            if loss_eval >= best:
                count += 1
                print('eval loss is not reduced for {} epoch'.format(count))
                print('best is {} until now'.format(best))
            else:
                count = 0
                print('eval loss is reduced from {:.5f} to {:.5f}, saving model'.format(best, loss_eval))
                self.save_model(chk_path)
                best = loss_eval

            if count == self.configs.patience:
                print('early stopping reached, best score is {:5f}'.format(best))
                break

    def save_configs(self, config_path):
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)

    def save_model(self, path):
        torch.save({'net': self.network.state_dict(),
                    'optimizer': self.opt.optimizer.state_dict()}, path)

class dataset_package(Dataset):
    def __init__(self, train_x, train_y):
        super().__init__()
        self.input = train_x
        self.target = train_y

    def GetDataShape(self):
        return {'input': self.input.shape,
                'target': self.target.shape}

    def __len__(self, ):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

########################################################################################################################

if __name__ == '__main__':
    print('Configs:\n', configs.__dict__)

    uv_train = np.load("data/975_uv_train.npy")
    zt_train = np.load("data/975_tz_train.npy")
    
    uv_val   = np.load("data/975_uv_test.npy")
    zt_val   = np.load("data/975_tz_test.npy")
    uv_train = np.concatenate((uv_train, zt_train), axis=1)
    del zt_train
    uv_val   = np.concatenate((uv_val, zt_val), axis=1)
    del zt_val
    ele = np.load('data/DEM_northeast.npy')
    

    ele[ele < 0] = 0
    ele= (ele - ele.mean()) / ele.std()

    print('processing training set')
    dataset_train = data_process(uv_train, samples_gap=3)
    del uv_train
    
    print('processing eval set')
    dataset_eval = data_process(uv_val, samples_gap=6)
    del uv_val

    train_x = dataset_train[:, :24, :, :, :]
    train_y = dataset_train[:, 24:, :, :, :]
    test_x = dataset_eval[:, :24, :, :, :]
    test_y = dataset_eval[:, 24:, :, :, :]
    
    dataset_train = dataset_package(train_x=train_x, train_y=train_y)
    dataset_test = dataset_package(train_x=test_x, train_y=test_y)
    del train_x, train_y, test_x, test_y
    print('Dataset_train Shape:\n', dataset_train.GetDataShape())
    print('Dataset_test Shape:\n', dataset_test.GetDataShape())

    trainer = Trainer(configs)
    trainer.save_configs('config_train.pkl')
    
    trainer.train(dataset_train, dataset_test, ele, 'chkfile/checkpoint.chk')