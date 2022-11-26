import torch

from torch import nn
from torchvision import transforms

data_dir = './data/classified_latents'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class LatentEncoder(nn.Module):
    def __init__(self, input_shape=(64, 512), num_classes=1):
        super(LatentEncoder, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Linear(input_shape[1], input_shape[1]),
            nn.ReLU(),
            nn.Linear(input_shape[1], input_shape[1]),
            nn.Tanh(),
            nn.Linear(input_shape[1], num_classes),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.encoder.apply(init_weights)

    def forward(self, x):
        return self.sigmoid(self.resnet(x))


model = LatentEncoder(input_shape=inp_shape).to(device)