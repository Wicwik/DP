from utils.L2FPipeline import L2FPipeline
from stylegan_generator import StyleGANGenerator
from models.MultilabelResnetClassifier import MultilabelResnetClassifier

import numpy as np
import h5py
import torch

import time

torch.set_printoptions(sci_mode=False)

network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl'
classifier_weights = '/home/robert/data/diploma-thesis/weights/classfier/resnet34_celeba10attr_10e.pt'

generator = StyleGANGenerator(network_pkl)
classifier = MultilabelResnetClassifier(n_classes=10)
classifier.load_state_dict(torch.load(classifier_weights))

pipeline = L2FPipeline(generator = generator, classifier = classifier)

dataset_len = 256000
batch_size = 64
n_batch = dataset_len/batch_size
z_path = '/home/robert/data/diploma-thesis/datasets/stylegan3/tpsi_1/latents/sample_z.h5'

z = None
with h5py.File(z_path, 'r') as f:
    z = np.split(f['z'][:], n_batch)

print('start')

for zi in z:
    start = time.time()
    pipeline.transform(zi)
    end = time.time()
    print(f'transform time: {end-start}')