import numpy as np
import torch

def weighted_rmse(y_pred,y_true):
    lat = np.linspace(38, 54, num=64, dtype=float, endpoint=True)
    RMSE = np.empty([y_pred.size(1)])
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    for i in range(y_pred.size(1)):
        RMSE[i] = np.sqrt(((y_pred[:,i,:,:] - y_true[:,i,:,:]).permute(0,2,1)**2*weights_lat).mean([-2,-1])).mean(axis=0)
    return RMSE

def weighted_mae(y_pred,y_true):
    lat = np.linspace(38, 54, num=64, dtype=float, endpoint=True)
    MAE = np.empty([y_pred.size(1)])
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    for i in range(y_pred.size(1)):
        MAE[i] = (abs(y_pred[:, i, :, :] - y_true[:, i, :, :]).permute(0, 2, 1) * weights_lat).mean([0, -2, -1])
    return MAE

def weighted_acc(y_pred,y_true):
    lat = np.linspace(38, 54, num=64, dtype=float, endpoint=True)
    ACC = np.empty([y_pred.size(1)])
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    w = torch.tensor(weights_lat)
    for i in range(y_pred.size(1)):
        clim = y_true[:,i,:,:].mean(0)
        a = y_true[:,i,:,:] - clim
        a_prime = (a - a.mean()).permute(0,2,1)
        fa = y_pred[:,i,:,:] - clim
        fa_prime = (fa - fa.mean()).permute(0,2,1)
        ACC[i] = (
                torch.sum(w * fa_prime * a_prime) /
                torch.sqrt(
                    torch.sum(w * fa_prime ** 2) * torch.sum(w * a_prime ** 2)
                )
        )
    return ACC

###输入预测值与真实值
y_pred = np.load(r'w_pred.npy')
y_pred = y_pred[:,:,None]
y_true = np.load(r'w_true.npy')
y_true = y_true[:,:,None]


for i in range(0,24,1):
    print(i+1)
    print('RMSE:', weighted_rmse(torch.tensor(y_pred[:,i,:,:,:]),torch.tensor(y_true[:,i,:,:,:])))
    print('MAE: ', weighted_mae(torch.tensor(y_pred[:,i,:,:,:]), torch.tensor(y_true[:,i,:,:,:])))
    print('ACC: ', weighted_acc(torch.tensor(y_pred[:,i,:,:,:]),torch.tensor(y_true[:,i,:,:,:])))
