import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

training_data = datasets.CelebA(root='data', train=True, download=True, transform=transform)
test_data = datasets.CelebA(root='data', train=False, download=True, transform=transform)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    inp_shape = X.shape
    break

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze())
plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
