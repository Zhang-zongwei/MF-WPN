import numpy as np
from torch.utils.data import Dataset
import xarray as xr
from pathlib import Path
import torch


def prepare_inputs_targets(len_time, input_gap, input_length, pred_shift, pred_length, samples_gap):
    assert pred_shift >= pred_length
    input_span = input_gap * (input_length - 1) + 1
    pred_gap = pred_shift // pred_length
    input_ind = np.arange(0, input_span, input_gap)
    target_ind = np.arange(0, pred_shift, pred_gap) + input_span + pred_gap - 1
    ind = np.concatenate([input_ind, target_ind]).reshape(1, input_length + pred_length)
    max_n_sample = len_time - (input_span+pred_shift-1)
    ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones((1, input_length+pred_length), dtype=int)
    return ind[::samples_gap]
   
class data_process(Dataset):
    def __init__(self, ossit, samples_gap):
        super().__init__()
        
        uv = []
        if ossit is not None:
            assert len(ossit.shape) == 4
            idx_uv = prepare_inputs_targets(ossit.shape[0], input_gap=1, input_length=24, pred_shift=24, pred_length=24, samples_gap=samples_gap)
            uv.append(ossit[idx_uv])

        self.uv = uv[0]

        assert self.uv.shape[1] == 48

    def GetDataShape(self):
        return {'uv': self.uv.shape}

    def __len__(self):
        return self.uv.shape[0]

    def __getitem__(self, idx):
        return self.uv[idx]