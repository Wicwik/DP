import torch
import time

from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision import transforms

from matplotlib import pyplot as plt

import numpy as np

transform = transforms.Compose([transforms.ToTensor()])

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
    print(f'Shape of X [N, C, H, W]: {X.shape}')
    print(f'Shape of y: {y.shape} {y.dtype}')
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

device = 'cpu' # cpu 500s, mps 15s
if torch.backends.mps.is_available():
    device = 'mps'
if torch.cuda.is_available():
    device = 'cuda'

print(f"Using {device} device")

class CelebAClassifier(nn.Module):
    def __init__(self, n_classes=5):
        super(CelebAClassifier, self).__init__()
    
        self.resnet = models.resnet34(weights='ResNet34_Weights.DEFAULT')
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=self.resnet.fc.in_features, out_features=n_classes)
        )

        self.sigmoid = nn.Sigmoid()
        # we do not init weitghts when using pretrained resnet
        # def init_weights(m):
        #     if type(m) in [nn.Linear, nn.Conv2d]:
        #         nn.init.kaiming_uniform_(m.weight)

        # self.resnet.apply(init_weights)

    def forward(self, x):
        return self.sigmoid(self.resnet(x))

model = CelebAClassifier(n_classes=5).to(device)
# print(model)

loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR = StepLR(optimizer, step_size=1, gamma=0.1)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    start = time.time()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)

        loss = loss_fn(pred, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            end = time.time()
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}] time: {end-start}')
            start = time.time()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            print(pred)
            test_loss += loss_fn(pred, y.float()).item()
            correct.append(torch.round(pred).eq(y).sum().cpu().numpy()/len(y[0])/len(y))
    test_loss /= num_batches
    print(f'Test Error: \n Accuracy: {(100*np.mean(correct)):>0.1f}%, Avg loss: {test_loss:>8f} \n')


epochs = 35
for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train(train_dataloader, model, loss_fn, optimizer)
    scheduler.step()
    test(test_dataloader, model, loss_fn)
print('Done!')

torch.save(model, 'Model_5attr_35eps')