import numpy as np


#####用于反归一化以及求取风速

uv = np.load(r'true.npy')

u = uv[:, :, 0, :, :] * 3.520639587404695 + 1.3793918561383813
v = uv[:, :, 1, :, :] * 4.020371225633169 - 0.1864951025062636

w = np.sqrt(u ** 2 + v ** 2)
np.save(r"true_w.npy", arr=w)


