from torch import nn

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_shape=(64, 512), num_classes=1):
        super(SimpleAutoencoder, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Linear(input_shape[1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024))
        
        self.decoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(512, input_shape[1]))  
        
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