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
valid_data = datasets.CelebA(root='data', split='valid', download=True, transform=transform)
test_data = datasets.CelebA(root='data', split='test', download=True, transform=transform)

names_data = training_data.attr_names
idx = [names_data.index('Eyeglasses')]

training_data.attr = training_data.attr[:,idx]
valid_data.attr = valid_data.attr[:, idx]
test_data.attr = test_data.attr[:, idx]

batch_size = 64
best_valid_loss = np.inf

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
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

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

print(f"Using {device} device")

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

model = CelebAClassifier(n_classes=len(idx)).to(device)

loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR = StepLR(optimizer, step_size=1, gamma=0.1)

train_loss_per_epoch = []
train_acc_per_epoch = []
valid_loss_per_epoch = []
valid_acc_per_epoch = []

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

# TODO refactor valid and test to single function
def valid(dataloader, model, loss_fn, save_filename=None):
    num_batches = len(dataloader)
    model.eval()
    valid_loss, correct = 0, []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            valid_loss += loss_fn(pred, y.float()).item()
            correct.append(torch.round(pred).eq(y).sum().cpu().numpy()/len(y[0])/len(y))

    valid_loss /= num_batches

    if valid_loss < best_valid_loss and save_filename:
        torch.save(model.state_dict(), save_filename)

    valid_acc = 100*np.mean(correct)
    print(f'Valid Error: \n Accuracy: {valid_acc:>0.1f}%, Avg loss: {valid_loss:>8f} \n')

    valid_loss_per_epoch.append(valid_loss)
    valid_acc_per_epoch.append(valid_acc)

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.float()).item()
            correct += torch.mean((torch.round(pred) == y).float())
    
    test_loss /= num_batches
    test_acc = 100*correct / num_batches
    print(f'Test Error: \n Accuracy: {test_acc:>0.1f}%, Avg loss: {test_loss:>8f} \n')

save_filename = 'resnet34_classifier_eyeglasses_5e.pt'
# epochs = 5
# for t in range(epochs):
#     print(f'Epoch {t+1}\n-------------------------------')
#     train(train_dataloader, model, loss_fn, optimizer)
#     scheduler.step()
#     valid(valid_dataloader, model, loss_fn, save_filename=save_filename)
# print('Done!')

print('Testing...')
model.load_state_dict(torch.load(save_filename))
test(test_dataloader, model, loss_fn)

# print('Plotting history...')
# plt.figure(figsize=(18, 4))
# plt.plot(train_loss_per_epoch, label = 'train')
# plt.plot(valid_loss_per_epoch, label = 'valid')
# plt.legend()
# plt.title('Loss Function')
# plt.show()

# plt.figure(figsize=(18, 4))
# plt.plot(train_acc_per_epoch, label = 'train')
# plt.plot(valid_acc_per_epoch, label = 'valid')
# plt.legend()
# plt.title('Accuracy')
# plt.show()

