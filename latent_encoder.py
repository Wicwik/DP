import torch
import time

from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

from latents_dataset import load as load_latents_dataset

data_path = 'data/generated_images/latents/sample_z.h5'
targets_path = 'data/predictions/predictions_resnet34_eyeglasses.pkl'

batch_size = 64
transform = transforms.Compose([])
target_transform = transforms.Compose([torch.round])
dataset = load_latents_dataset(data_path, targets_path, transform=transform, target_transform=target_transform)
train_data, valid_data, test_data = random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

train_loss_per_epoch, train_acc_per_epoch = [], []

for X, y in test_dataloader:
    print(f'Shape of X [N, C, H, W]: {X.shape}')
    print(f'Shape of y: {y.shape} {y.dtype}')
    inp_shape = X.shape
    print(y)
    break

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class LatentEncoder(nn.Module):
    def __init__(self, input_shape=(64, 512), num_classes=1):
        super(LatentEncoder, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Linear(input_shape[1], input_shape[1]),
            nn.ReLU(),
            nn.Linear(input_shape[1], input_shape[1]),
            nn.Tanh(),
            nn.Linear(input_shape[1], num_classes),
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

input_shape = (batch_size, len(dataset[0][0]))
print(input_shape)
model = LatentEncoder(input_shape=input_shape).to(device)

print(model)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR = StepLR(optimizer, step_size=1, gamma=0.1)

epochs = 5
for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train(train_dataloader, model, loss_fn, optimizer)
    scheduler.step()
    # valid(valid_dataloader, model, loss_fn, save_filename=save_filename)
print('Done!')