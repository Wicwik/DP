import torch

from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision import transforms

from matplotlib import pyplot as plt

import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

training_data = datasets.CelebA(root='data', split='train', download=True, transform=transform)
test_data = datasets.CelebA(root='data', split='test', download=True, transform=transform)

names_data = training_data.attr_names
idx = [names_data.index('Eyeglasses'), names_data.index('Young'), names_data.index('No_Beard'), names_data.index('Smiling'), names_data.index('Male')]

training_data.attr = training_data.attr[:,idx]
test_data.attr = test_data.attr[:, idx]

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
    img, _ = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    plt.imshow(img.squeeze())
plt.show()

# device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class CelebAClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(CelebAClassifier, self).__init__()
    
        self.resnet = models.resnet34(pretrained=True)
        self.resnet_no_fc = nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.classifier.apply(init_weights)
        self.resnet_no_fc.apply(init_weights)

    def forward(self, x):
        x = self.resnet_no_fc(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = CelebAClassifier(num_classes=5).to(device)
print(model)

for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data.shape)

loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR = StepLR(optimizer, step_size=1, gamma=0.1)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)

        loss = loss_fn(pred, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.float()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    scheduler.step()
    test(test_dataloader, model, loss_fn)
print("Done!")