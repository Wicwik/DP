from torch import nn

class LatentEncoder(nn.Module):
    def __init__(self, input_shape=(64, 512), num_classes=1):
        super(LatentEncoder, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 25))
        
        self.decoder = nn.Sequential(
            nn.Linear(257, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Tanh())        

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.encoder.apply(init_weights)

    def forward(self, x, y):
        latent = self.encoder(x)
        x = self.decoder(torch.cat((y.reshape((y.shape[0],-1)), latent),1))
        return x,latent