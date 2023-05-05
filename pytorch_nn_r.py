import torch
import time
import pickle

import numpy as np

from torchmetrics import MeanSquaredLogError
from torchmetrics import MeanAbsoluteError
from torchmetrics import R2Score

from torch.utils.data import DataLoader, TensorDataset, random_split

from memory_profiler import profile

class TorchRegressionTraining():
    __model = None
    __train = None
    __valid= None
    __test = None
    __loss_fn = None
    __optimizer = None
    __save_filename = None
    __device = 'cpu'

    __best_valid_loss = np.inf
    __mae = MeanAbsoluteError()
    __r2 = R2Score(num_outputs=512, multioutput='uniform_average')

    __experiment = None
    __epoch = None
   
    __loss_per_epoch, __mae_per_epoch, __r2_per_epoch  = {}, {}, {}

    __loss_per_epoch['train'], __mae_per_epoch['train'], __r2_per_epoch['train']  = [], [], []
    __loss_per_epoch['valid'], __mae_per_epoch['valid'], __r2_per_epoch['valid'] = [], [], []
    
    @profile
    def __init__(self, model, pkl_path, loss_fn, optimizer, batch_size, save_filename = None, experiment = None):
        self._set_device()
        print(self.__device)

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        dataset = TensorDataset(torch.Tensor(np.stack(data[:, 0])), torch.Tensor(np.stack(data[:, 1])), torch.Tensor(np.stack(data[:, 2])))

        train_data, valid_data, test_data = random_split(dataset, [0.8, 0.1, 0.1])

        self.__batch_size = batch_size
        self.__train = DataLoader(train_data, batch_size=self.__batch_size, shuffle=True)
        self.__valid = DataLoader(valid_data, batch_size=self.__batch_size, shuffle=True)
        self.__test = DataLoader(test_data, batch_size=self.__batch_size, shuffle=False)
        
        self.__model = model.to(self.__device)
        self.__loss_fn = loss_fn
        self.__optimizer = optimizer
        self.__save_filename = save_filename
        self.__experiment = experiment

        if self.__experiment:
            self.__experiment.log_parameters({'steps': len(self.__train)})
        
        np.set_printoptions(formatter={'float_kind':"{:.6f}".format})
        
    def _set_device(self):
        if torch.cuda.is_available():
            self.__device = 'cuda'
        
    @profile
    def _train(self):
        dataloader = self.__train
        
        train_loss, train_mae, train_r2 = 0, 0, 0
        num_batches = len(dataloader)
        size = len(dataloader.dataset)
        
        self.__model.train()
        start = time.time()
        batch = 0
        for X, features, y in dataloader:
            X, features, y = X.to(self.__device), features.to(self.__device), y.to(self.__device)

            pred = self.__model(X, features)

            loss = self.__loss_fn(pred, y.float())
            
            pred, y = pred.cpu(), y.cpu()
            mae = self.__mae(pred, y).detach().numpy()
            r2 = self.__r2(pred, y).detach().numpy()
            
            train_loss += loss.item()
            train_mae += mae
            train_r2 += r2

            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            if batch % 500 == 0:
                end = time.time()
                if self.__experiment:
                    metrics = {'train_loss': loss.item(), 'train_mae': mae, 'train_r2': r2}
                    self.__experiment.log_metrics(metrics, step=batch)

                loss, current = loss.item(), batch * len(X)
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}] time: {end-start} mae: {mae} r2: {r2}')
                start = time.time()
            batch += 1

        train_loss /= num_batches
        train_mae /= num_batches
        train_r2 /= num_batches

        self.__loss_per_epoch['train'].append(train_loss)
        self.__mae_per_epoch['train'].append(train_mae)
        self.__r2_per_epoch['train'].append(train_r2)

        if self.__experiment:
            metrics = {'avg_train_loss_per_epoch': train_loss, 
                       'avg_train_mae_per_epoch': train_mae, 
                       'avg_train_r2_per_epoch': train_r2
                       }
            self.__experiment.log_metrics(metrics, step=self.__epoch)

    @profile
    def _valid(self, test = False):
        dataloader = self.__valid
        print_keyword = 'Valid'
        
        if test:
            dataloader = self.__test
            print_keyword = 'Test'
        
        valid_loss, valid_mae, valid_r2 = 0, 0, 0
        num_batches = len(dataloader)
        
        self.__model.eval()
        batch = 0
        with torch.no_grad():
            for X, features, y in dataloader:
                X, features, y = X.to(self.__device), features.to(self.__device), y.to(self.__device)
                pred = self.__model(X, features)

                
                loss = self.__loss_fn(pred, y.float())

                pred, y = pred.cpu(), y.cpu()
                mae = self.__mae(pred, y).detach().numpy()
                r2 = self.__r2(pred, y).detach().numpy()
                
                if self.__experiment:
                    metrics = {'valid_loss': loss, 'valid_mae': mae, 'valid_r2': r2,}
                    self.__experiment.log_metrics(metrics, step=batch)

                valid_loss += loss.item()
                valid_mae += mae
                valid_r2 += r2

                batch += 1

        valid_loss /= num_batches
        valid_mae /= num_batches
        valid_r2 /= num_batches

        if self.__experiment:
            metrics = {f'avg_{print_keyword.lower()}_loss_per_epoch': valid_loss, 
                       f'avg_{print_keyword.lower()}_mae_per_epoch': valid_mae, 
                       f'avg_{print_keyword.lower()}_r2_per_epoch': valid_r2
                       }
            if test:
                self.__experiment.log_metrics(metrics)
            else:
                self.__experiment.log_metrics(metrics, step=self.__epoch)
            

        print(f'{print_keyword} | Error: \n MAE: {valid_mae:>8f}, R^2: {valid_r2:>8f}, Avg loss: {valid_loss:>8f} \n')

        if not test:
            if valid_loss < self.__best_valid_loss and self.__save_filename:
                print(f'Saving model to file: {self.__save_filename}')
                torch.save(self.__model.state_dict(), self.__save_filename)
                self.__best_valid_loss = valid_loss

            self.__loss_per_epoch['valid'].append(valid_loss)
            self.__mae_per_epoch['valid'].append(valid_mae)
            self.__r2_per_epoch['valid'].append(valid_r2)

    
    def run_training(self, epochs = 10):
        for t in range(epochs):
            print(f'Epoch {t+1}\n-------------------------------')
            self.__epoch = t+1
            self.__experiment.set_epoch(self.__epoch)
            self._train()
            self._valid()
        print('Done!')

        if self.__save_filename:
            self.load_weights(self.__save_filename)
        
        self._valid(test=True)
        
    def load_weights(self, save_filename):
        self.__model.load_state_dict(torch.load(save_filename))
    
    def predict(self, x):
        with torch.no_grad():
            x = x.to(self.__device)
            pred = self.__model(x)

        return pred.cpu()
    
    def plot_metrics(self):
        pass
