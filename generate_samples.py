import os

import dnnlib
import PIL.Image
import torch
import numpy as np

from typing import List, Optional, Tuple, Union

import legacy

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl'

device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as f:
	G = legacy.load_network_pkl(f)['G_ema'].to(device)

label = torch.zeros([1, G.c_dim], device=device)

translate = (0,0)
rotate = 0
truncation_psi = 1
noise_mode = 'const'
n_samples = 10
outdir = 'static/samples'

os.makedirs(outdir, exist_ok=True)

for idx in range(n_samples):
	z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)

	if hasattr(G.synthesis, 'input'):
		m = make_transform(translate, rotate)
		m = np.linalg.inv(m)
		G.synthesis.input.transform.copy_(torch.from_numpy(m))

	img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
	img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
	PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/image{idx:04d}.png')