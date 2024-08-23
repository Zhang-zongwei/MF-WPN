import numpy as np


#####用于反归一化以及求取风速

uv_true = np.load(r'uv_true.npy')
uv_pred = np.load(r'uv_pred.npy')

u_true = uv_true[:, :, 0, :, :] * 3.520639587404695 + 1.3793918561383813
v_true = uv_true[:, :, 1, :, :] * 4.020371225633169 - 0.1864951025062636

u_pred = uv_pred[:, :, 0, :, :] * 3.520639587404695 + 1.3793918561383813
v_pred = uv_pred[:, :, 1, :, :] * 4.020371225633169 - 0.1864951025062636

w_true = np.sqrt(u_true ** 2 + v_true ** 2)
w_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

np.save(r"w_pred.npy", arr=w_pred)
np.save(r"w_true.npy", arr=w_true)


