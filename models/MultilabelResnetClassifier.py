from torch import nn
from torchvision import models

class MultilabelResnetClassifier(nn.Module):
    def __init__(self, n_classes=5):
        super(MultilabelResnetClassifier, self).__init__()
    
        self.resnet = models.resnet34(weights='DEFAULT')
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=self.resnet.fc.in_features, out_features=n_classes)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.resnet(x))