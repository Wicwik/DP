import dnnlib
import PIL.Image
import torch
import numpy as np

import legacy

network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl'

device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as f:
	G = legacy.load_network_pkl(f)['G_ema'].to(device)

label = torch.zeros([1, G.c_dim], device=device)

z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)

if hasattr(G.synthesis, 'input'):
	print('input')
	m = make_transform(translate, rotate)
	m = np.linalg.inv(m)
	G.synthesis.input.transform.copy_(torch.from_numpy(m))