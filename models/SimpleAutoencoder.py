from torch import nn

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_shape=(64, 512), num_classes=1):
        super(SimpleAutoencoder, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Linear(input_shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_shape[1]),
        )
        
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

        self.tanh = nn.Tanh()

    def forward(self, x, _):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return self.tanh(decoded)