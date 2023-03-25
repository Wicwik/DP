import os

import stylegan3.dnnlib as dnnlib
import PIL.Image
import torch
import numpy as np
import h5py
import cv2

from tqdm import tqdm
from tqdm.contrib.telegram import tqdm as ttqdm, trange

from typing import List, Optional, Tuple, Union

import stylegan3.legacy as legacy

class StyleGANGenerator:
	def __init__(self, network_pkl):
		self.device = torch.device('cuda')
		self.network_pkl = network_pkl
		self.G = self._get_sylegan_network(self.network_pkl)

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
		

	def generate_images_from_z(self, z_path, n_batch, batch_size, outdir, truncation_psi = 1, noise_mode = 'const', translate = (0,0), rotate = 0, img_size = (178, 218)):
		label = torch.zeros([batch_size, self.G.c_dim], device=self.device)
		
		os.makedirs(outdir + '/imgs/noclass', exist_ok=True)
		os.makedirs(outdir + '/latents', exist_ok=True)

		z = None
		with h5py.File(z_path, 'r') as f:
			z = np.split(f['z'][:], n_batch)

		for i in trange(n_batch, token='5014943200:AAE9WepCFlwI-4M9kBxcflezF36s2YUoTYo', chat_id='528072721'):
			z_batch = torch.from_numpy(z[i]).to(self.device)

			if hasattr(self.G.synthesis, 'input'):
				m = self._make_transform(translate, rotate)
				m = np.linalg.inv(m)
				self.G.synthesis.input.transform.copy_(torch.from_numpy(m))

			imgs = self.G(z_batch, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
			imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
			imgs = np.asarray([cv2.resize(img.cpu().numpy(), dsize=img_size, interpolation=cv2.INTER_LANCZOS4) for img in imgs])

			for j, img in enumerate(imgs):
				PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/imgs/noclass/image{i*batch_size+j:06d}.png')


	def generate_images(self, n_batch, batch_size, outdir, truncation_psi = 1, noise_mode = 'const', translate = (0,0), rotate = 0, img_size = (178, 218)):
		z_list = []

		# network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl'
		# network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl' 

		# used for conditional generation
		label = torch.zeros([batch_size, self.G.c_dim], device=self.device)
		
		os.makedirs(outdir + '/imgs/noclass', exist_ok=True)
		os.makedirs(outdir + '/latents', exist_ok=True)

		for i in trange(n_batch, token='5014943200:AAE9WepCFlwI-4M9kBxcflezF36s2YUoTYo', chat_id='528072721'):
			z = np.random.randn(batch_size, self.G.z_dim)
			z_list.append(z)

			z = torch.from_numpy(z).to(self.device)
			
			if hasattr(self.G.synthesis, 'input'):
				m = self._make_transform(translate, rotate)
				m = np.linalg.inv(m)
				self.G.synthesis.input.transform.copy_(torch.from_numpy(m))

			imgs = self.G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
			imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
			imgs = np.asarray([cv2.resize(img.cpu().numpy(), dsize=img_size, interpolation=cv2.INTER_LANCZOS4) for img in imgs])

			for j, img in enumerate(imgs):
				PIL.Image.fromarray(img, 'RGB').save(f'{outdir}imgs/noclass/image{i*batch_size+j:06d}.png')

		z_concat = np.concatenate(z_list, axis=0)
		print('Array of latent vectors shape:',z_concat.shape)

		with h5py.File(outdir + '/latents/sample_z.h5', 'w') as f:
			f.create_dataset('z', data=z_concat)


	def generate_one(self, truncation_psi = 1, noise_mode = 'const', translate = (0,0), rotate = 0):
		label = torch.zeros([1, self.G.c_dim], device=self.device)

		z = torch.from_numpy(np.random.randn(1, self.G.z_dim)).to(self.device)

		if hasattr(self.G.synthesis, 'input'):
			m = self._make_transform(translate, rotate)
			m = np.linalg.inv(m)
			self.G.synthesis.input.transform.copy_(torch.from_numpy(m))

		img = self.G(z, None, truncation_psi=truncation_psi, noise_mode=noise_mode)
		img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
		return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

	def generate_from(self, z, filepath = None, truncation_psi = 1, noise_mode = 'const', translate = (0,0), rotate = 0):
		label = torch.zeros([len(z), self.G.c_dim], device=self.device)

		z = torch.from_numpy(z).to(self.device)

		if hasattr(self.G.synthesis, 'input'):
			m = self._make_transform(translate, rotate)
			m = np.linalg.inv(m)
			self.G.synthesis.input.transform.copy_(torch.from_numpy(m))

		imgs = self.G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
		imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

		if filepath:
			for i, img in enumerate(imgs):
				PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(f'{filepath}_{i:06d}.png')
		else:
			return imgs.cpu().numpy()

	def get_random_vectors(self, n):
		return np.random.randn(n, self.G.z_dim)


	def test_generated_images(self, outdir, n, truncation_psi = 1, noise_mode = 'const'):
		with h5py.File(outdir + '/latents/sample_z.h5', 'r') as f:
			z = f['z'][:]

		z = np.reshape(z[n], (1,512))
		z = torch.from_numpy(z).to(self.device)

		img = self.G(z, None, truncation_psi=truncation_psi, noise_mode=noise_mode)
		img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

		PIL.Image.fromarray(img, 'RGB').save('test.png')