import torch
from torch import nn

class LatentFeatureExtractor(nn.Module):
    def __init__(self, input_shape=(64, 512), n_classes=1):
        super(LatentFeatureExtractor, self).__init__()
    
        vector_shape = input_shape[1]
        self.extractor = nn.Sequential(
            nn.Linear(n_classes+vector_shape, 1024),
            nn.ReLU(),
            nn.Linear(1024, vector_shape),
            nn.Tanh())
        
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.extractor.apply(init_weights)

    def forward(self, x, y):
        print(torch.cat((y.reshape((y.shape[0],-1)), x),1))
        return self.extractor(torch.cat((y.reshape((y.shape[0],-1)), x),1))
