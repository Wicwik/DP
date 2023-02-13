import torch

def norm_zero_one(t):
    mean, std = torch.mean(t), torch.std(t)

    return (t-mean)/std