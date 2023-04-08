import torch
from torch import nn
from torchvision import transforms

import stylegan3.dnnlib as dnnlib
import stylegan3.legacy as legacy
from models.MultilabelResnetClassifier import MultilabelResnetClassifier

import cv2

class LatentFeatureExtractor(nn.Module):
    def __init__(self, input_shape=(64, 512), n_classes=1, network_pkl='https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl', classifier_weights='/home/robert/data/diploma-thesis/weights/classfier/resnet34_celeba10attr_10e.pt', tpsi=1, noise_mode = 'const'):
        super(LatentFeatureExtractor, self).__init__()

        self.img_size = (218, 178)
        self.tpsi = tpsi
        self.noise_mode = noise_mode
        self.normalize = transforms.Normalize(mean=0, std=1)
    
        vector_shape = input_shape[1]

        self.extractor = nn.Sequential(
            nn.Linear(vector_shape+n_classes, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
      )
        
        self.generator = None
        with dnnlib.util.open_url(network_pkl) as f:
            self.generator = legacy.load_network_pkl(f)['G_ema']

        self.label = torch.zeros([input_shape[0], self.generator.c_dim])

        for param in self.generator.parameters():
            param.requires_grad = False

        self.classifier = MultilabelResnetClassifier(n_classes=10)
        self.classifier.load_state_dict(torch.load(classifier_weights))

        for param in self.classifier.parameters():
            param.requires_grad = False
        
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.extractor.apply(init_weights)

    def forward(self, x, y):
        extracted = self.extractor(torch.cat((x, y),dim=1))
        imgs = self.generator(x + extracted, self.label, truncation_psi=self.tpsi, noise_mode=self.noise_mode)
        imgs = transforms.Resize(size=self.img_size)(imgs)

        return self.classifier(imgs)
