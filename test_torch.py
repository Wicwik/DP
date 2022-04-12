import torch
print('Current device:', torch.cuda.get_device_name(torch.cuda.current_device()))

from torch.utils.cpp_extension import CUDA_HOME
print(CUDA_HOME)

#import os
#print(os.environ.get('CUDA_PATH'))

#model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
#model.eval()
