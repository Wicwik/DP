import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

import pickle
import os

n_classes = 1
save_filename = 'resnet34_classifier_eyeglasses_5e.pt'
path_to_data = '/home/robert/data/diploma-thesis/datasets/stylegan3/tpsi_1/imgs'

transform = transforms.Compose([transforms.ToTensor()])

dataset = datasets.ImageFolder(path_to_data, transform=transform)

batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

np.set_printoptions(formatter={'float_kind':"{:.6f}".format})

for X, y in dataloader:
    print(f'Shape of X [N, C, H, W]: {X.shape}')
    print(f'Shape of y: {y.shape} {y.dtype}')
    inp_shape = X.shape
    break

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    img, _ = dataset[i-1]
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    plt.imshow(img.squeeze())
plt.show()

class CelebAClassifier(nn.Module):
    def __init__(self, n_classes=5):
        super(CelebAClassifier, self).__init__()
    
        self.resnet = models.resnet34(weights='DEFAULT')
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=self.resnet.fc.in_features, out_features=n_classes)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.resnet(x))

def predict(model, dataloader):
    preds = np.array([])
    model.eval()
    with torch.no_grad():
        for X, _ in tqdm(dataloader):
            X = X.to(device)
            preds = np.concatenate((preds, model(X).cpu().numpy()), axis=None)

    return preds

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

print(f"Using {device} device")

model = CelebAClassifier(n_classes=n_classes).to(device)
model.load_state_dict(torch.load(save_filename))

preds = predict(model=model, dataloader=dataloader)

os.makedirs('./data/predictions',exist_ok=True)
with open('data/predictions/predictions_resnet34_eyeglasses.pkl', 'wb') as f:
    pickle.dump(preds, f)
