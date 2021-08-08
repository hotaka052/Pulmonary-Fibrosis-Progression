import numpy as np

import torch

def score(y_true, y_pred, device):
    """
    評価指標を再現した関数
    """
    c1 = torch.ones([len(y_true)], dtype = torch.float32, device = device) * 70.0
    c2 = torch.ones([len(y_true)], dtype = torch.float32, device = device) * 1000.0
    
    sigma = y_pred[:,2] - y_pred[:,0]
    fvc_pred = y_pred[:,1]
    
    sigma_clip = torch.max(sigma, c1)
    delta = torch.abs(y_true - fvc_pred)
    delta = torch.min(delta, c2)
    sq2 = torch.sqrt(torch.tensor(2, dtype = torch.float32, device = device))
    metric = (delta / sigma_clip) * sq2 + torch.log(sigma_clip * sq2)
    
    return torch.mean(metric)

def qloss(y_true, y_pred, device):
    y_true = y_true.repeat(3, 1)
    y_true = torch.transpose(y_true, 1, 0)
    qs = [0.2, 0.50, 0.8]
    q = torch.tensor(np.array([qs]), dtype = torch.float32, device = device)
    e = y_true - y_pred
    v = torch.max(q * e, (q-1) * e)
    return torch.mean(v)

def mloss(_lambda, device):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred, device) + (1 - _lambda) * score(y_true, y_pred, device)
    return loss