import torch
from torch import nn

class LatentsAutoencoder(nn.Module):
    def __init__(self, input_shape=(64, 512), num_classes=1):
        super(LatentsAutoencoder, self).__init__()
    
        vector_shape = input_shape[1]
        self.encoder = nn.Sequential(
            nn.Linear(vector_shape, vector_shape),
            nn.Tanh(),
            nn.Linear(vector_shape, vector_shape),
            nn.Tanh(),
            nn.Linear(vector_shape, vector_shape),
            nn.Tanh(),
            nn.Linear(vector_shape, vector_shape),
            nn.Tanh(),
            nn.Linear(vector_shape, vector_shape),
        )

        self.decoder = nn.Sequential(
            nn.Linear(num_classes+vector_shape, vector_shape),
            nn.Tanh(),
            nn.Linear(vector_shape, vector_shape),
            nn.Tanh(),
            nn.Linear(vector_shape, vector_shape),
            nn.Tanh(),
            nn.Linear(vector_shape, vector_shape),
            nn.Tanh(),
            nn.Linear(vector_shape, vector_shape),
            nn.Tanh()
        )

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

    def forward(self, x, y):
        encoded = self.encoder(x)
        decoded = self.decoder(torch.cat((y, encoded),1))
        return decoded