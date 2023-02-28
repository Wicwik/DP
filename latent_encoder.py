import torch
import time

import numpy as np

from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

from torchmetrics.classification import BinaryPrecision
precision = BinaryPrecision()

from torchmetrics.classification import BinaryRecall
recall = BinaryRecall()

from latents_dataset import load_custom_dataset as load_latents_dataset

from utils.ops import norm_zero_one

data_path = '/home/robert/data/diploma-thesis/datasets/stylegan3/tpsi_1/latents/sample_z.h5'
targets_path = '/home/robert/data/diploma-thesis/predictions/stylegan3/tpsi_1/resnet34_eyeglasses.pkl'

batch_size = 64

transform = transforms.Compose([])
target_transform = transforms.Compose([torch.round])
dataset = load_latents_dataset(data_path, targets_path, transform=transform, target_transform=target_transform)
train_data, valid_data, test_data = random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

train_loss_per_epoch, train_acc_per_epoch = [], []
valid_loss_per_epoch, valid_acc_per_epoch = [], []

# the noise is generated from normal dist. therefore there is already a mean 0 and variance 1

for X, y in test_dataloader:
    print(f'Shape of X [N, C, H, W]: {X.shape}')
    print(f'Shape of y: {y.shape} {y.dtype}')
    break

device = 'cpu'
if torch.cuda.is_available( ):
    device = 'cuda'

class LatentEncoder(nn.Module):
    def __init__(self, input_shape=(64, 512), num_classes=1):
        super(LatentEncoder, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Linear(input_shape[1], 512),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

        self.sigmoid = nn.Sigmoid()

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.encoder.apply(init_weights)

    def forward(self, x):
        return self.sigmoid(self.encoder(x))

def train(dataloader, model, loss_fn, optimizer):
    train_loss, train_acc = 0, 0
    num_batches = len(dataloader)

    size = len(dataloader.dataset)
    model.train()
    start = time.time()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)

        loss = loss_fn(pred, y.float())
        train_loss += loss.item()

        acc = torch.round(pred).eq(y).sum().cpu().numpy()/len(y[0])/len(y)
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            end = time.time()
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}] time: {end-start} acc: {acc}')
            start = time.time()

    train_loss /= num_batches
    train_acc /= num_batches

    train_loss_per_epoch.append(train_loss)
    train_acc_per_epoch.append(train_acc)

def valid(dataloader, model, loss_fn, save_filename=None):
    num_batches = len(dataloader)
    model.eval()
    valid_loss, valid_acc, valid_precision, valid_recall = 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            valid_loss += loss_fn(pred, y.float()).item()
            valid_acc += torch.round(pred).eq(y).sum().cpu().numpy()/len(y[0])/len(y)
            valid_precision += precision(pred.cpu(), y.cpu())
            valid_recall += recall(pred.cpu(), y.cpu())

    valid_loss /= num_batches
    valid_acc /= num_batches
    valid_precision /= num_batches
    valid_recall /= num_batches

    print(f'Valid Error: \n Accuracy: {valid_acc:>8f}, Precision: {valid_precision:>8f}, Recall: {valid_recall:>8f}, Avg loss: {valid_loss:>8f} \n')

    valid_loss_per_epoch.append(valid_loss)
    valid_acc_per_epoch.append(valid_acc)


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.float()).item()
            test_acc += torch.round(pred).eq(y).sum().cpu().numpy()/len(y[0])/len(y)
    
    test_loss /= num_batches
    test_acc /= num_batches
    print(f'Test Error: \n Accuracy: {test_acc:>0.8f}, Avg loss: {test_loss:>8f} \n')

input_shape = (batch_size, len(dataset[0][0]))
model = LatentEncoder(input_shape=input_shape).to(device)

print(model)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = StepLR = StepLR(optimizer, step_size=1, gamma=0.1)

epochs = 20
for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train(train_dataloader, model, loss_fn, optimizer)
    # scheduler.step()
    valid(valid_dataloader, model, loss_fn)
print('Done!')

test(test_dataloader, model, loss_fn)