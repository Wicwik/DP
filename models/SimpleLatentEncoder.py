import torch
from torch import nn

class LatentsAutoencoder(nn.Module):
    def __init__(self, input_shape=(64, 512), num_classes=1):
        super(LatentsAutoencoder, self).__init__()
    
        vector_shape = input_shape[1]
        self.encoder = nn.Sequential(
            nn.Linear(vector_shape, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024))
        
        self.decoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, vector_shape),
            nn.Tanh())   

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x,latent
