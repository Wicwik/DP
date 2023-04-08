from utils.L2FPipeline import L2FPipeline
from stylegan_generator import StyleGANGenerator
from models.MultilabelResnetClassifier import MultilabelResnetClassifier

import numpy as np
import h5py
import torch

import time

import matplotlib.pyplot as plt

torch.set_printoptions(sci_mode=False)
np.set_printoptions(formatter={'float_kind':"{:.6f}".format})

network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl'
classifier_weights = '/home/robert/data/diploma-thesis/weights/classfier/resnet34_celeba10attr_10e.pt'

generator = StyleGANGenerator(network_pkl)
classifier = MultilabelResnetClassifier(n_classes=10)
classifier.load_state_dict(torch.load(classifier_weights))

pipeline = L2FPipeline(generator = generator, classifier = classifier, tpsi=0.7)

dataset_len = 256000
batch_size = 64
n_batch = dataset_len/batch_size
z_path = '/home/robert/data/diploma-thesis/datasets/stylegan3/tpsi_1/latents/sample_z.h5'

z = None
with h5py.File(z_path, 'r') as f:
    z = np.split(f['z'][:], n_batch)

print('start')
print(['Attractive', 'Eyeglasses', 'No_Beard', 'Male', 'Black_Hair', 'Blond_Hair', 'Mustache', 'Young', 'Smiling', 'Bald'])

for zi in z:
    start = time.time()
    preds, imgs = pipeline.transform(zi)
    end = time.time()
    print(f'transform time: {end-start}')

    imgs = np.transpose(imgs, (0,2,3,1))

    rows = 2
    cols = 3
    f, ax = plt.subplots(rows, cols)

    imgs_i = [0,6,30,25,36,57]

    for i in range(rows):
        for j in range(cols):
            ax[i, j].imshow(imgs[imgs_i[i*cols + j]])
            print(imgs_i[i*cols + j], preds[imgs_i[i*cols + j]].numpy())
            ax[i, j].axis('off')
            ax[i, j].set_title(chr(ord('a') + i*cols + j))


    plt.show()

    exit()