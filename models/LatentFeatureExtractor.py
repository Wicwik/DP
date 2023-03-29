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

        self.tpsi = tpsi
        self.noise_mode = noise_mode
        self.to_tensor = transforms.ToTensor()
        self.label = torch.zeros([input_shape, self.generator.c_dim])
    
        vector_shape = input_shape[1]
        self.extractor = nn.Sequential(
            nn.Linear(n_classes+vector_shape, 1024),
            nn.ReLU(),
            nn.Linear(1024, vector_shape),
            nn.Tanh())
        
        with dnnlib.util.open_url(network_pkl) as f:
            self.generator = legacy.load_network_pkl(f)['G_ema']

        self.classifier = MultilabelResnetClassifier(n_classes=10)
        self.classifier.load_state_dict(torch.load(classifier_weights))
        
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.extractor.apply(init_weights)

    def forward(self, x, y):
        extracted = self.extractor(torch.cat((y.reshape((y.shape[0],-1)), x),1))
        imgs = self.generator(extracted, self.label, truncation_psi=self.tpsi, noise_mode=self.noise_mode)
        imgs = torch.stack([self.to_tensor(cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_LANCZOS4)).to(self.device) for img in imgs])

        return self.classifier(imgs)
