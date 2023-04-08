import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

import pickle
import os

n_classes = 10
save_filename = '/home/robert/data/diploma-thesis/weights/classfier/resnet34_celeba10attr_10e.pt'
path_to_data = '/home/robert/data/diploma-thesis/datasets/stylegan2/tpsi_1/imgs'

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

from models.MultilabelResnetClassifier import MultilabelResnetClassifier

def predict(model, dataloader):
    preds = np.array([])
    model.eval()
    with torch.no_grad():
        for X, _ in tqdm(dataloader):
            X = X.to(device)
            if preds.size == 0:
                preds = model(X).cpu().numpy()
            else:
                preds = np.concatenate((preds, model(X).cpu().numpy()), axis=0)

    return preds

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

print(f"Using {device} device")

model = MultilabelResnetClassifier(n_classes=n_classes).to(device)
model.load_state_dict(torch.load(save_filename))

preds = predict(model=model, dataloader=dataloader)

preds_path = '/home/robert/data/diploma-thesis/predictions/stylegan2/tpsi_1/resnet34_10attr.pkl'
os.makedirs('/home/robert/data/diploma-thesis/predictions/stylegan2/tpsi_1',exist_ok=True)
with open(preds_path, 'wb') as f:
    pickle.dump(preds, f)
