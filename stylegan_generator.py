import os

import dnnlib
import PIL.Image
import torch
import numpy as np

from tqdm import tqdm

from typing import List, Optional, Tuple, Union

import legacy

class StyleGANGenerator:
	def __init__(self, network_pkl):
		self.device = torch.device('cuda')
		self.network_pkl = network_pkl
		self.G = self._get_sylegan_network(self.network_pkl)
		print(self.G.c_dim)

	def _make_transform(self, translate: Tuple[float,float], angle: float):
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

	def _get_sylegan_network(self, network_pkl):
		with dnnlib.util.open_url(network_pkl) as f:
			return legacy.load_network_pkl(f)['G_ema'].to(self.device)

	def generate_images(self, n_samples, outdir, truncation_psi = 1, noise_mode = 'const', translate = (0,0), rotate = 0):
		 

		# network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl'
		# network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl' 

		# used for conditional generation
		label = torch.zeros([1, self.G.c_dim], device=self.device)
		
		os.makedirs(outdir, exist_ok=True)

		for idx in tqdm(range(n_samples)):
			z = torch.from_numpy(np.random.randn(1, self.G.z_dim)).to(self.device)

			if hasattr(self.G.synthesis, 'input'):
				m = self._make_transform(translate, rotate)
				m = np.linalg.inv(m)
				self.G.synthesis.input.transform.copy_(torch.from_numpy(m))

			img = self.G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
			img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
			PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/image{idx:04d}.png')
