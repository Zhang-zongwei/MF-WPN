import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

configs.n_cpu = 0
configs.device = torch.device('cuda:0')
configs.batch_size_test = 4
configs.batch_size = 4
configs.weight_decay = 0
configs.display_interval = 100
configs.num_epochs = 100
configs.early_stopping = True
configs.patience = 10
configs.gradient_clipping = False
configs.clipping_threshold = 1.
configs.warmup = 5000
configs.d_model = 256
configs.input_gap = 1
configs.pred_shift = 24
configs.dropout = 0.2
configs.ssr_decay_rate = 5.e-5
