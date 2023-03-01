from torch import nn

class LatentEncoder(nn.Module):
    def __init__(self, input_shape=(64, 512), num_classes=1):
        super(LatentEncoder, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Linear(input_shape[1], 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.125),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.125),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

        self.sigmoid = nn.Sigmoid()

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.encoder.apply(init_weights)

    def forward(self, x):
        return self.sigmoid(self.encoder(x))