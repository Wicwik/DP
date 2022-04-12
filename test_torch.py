import torch

print('Current device:', torch.cuda.get_device_name(torch.cuda.current_device()))
