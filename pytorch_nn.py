import torch
import time

import numpy as np

from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryAccuracy

class NNUtil():
    __model = None
    __train = None
    __valid= None
    __test = None
    __loss_fn = None
    __optimizer = None
    __save_filename = None
    __device = 'cpu'
    __best_valid_loss = np.inf
    __precision = BinaryPrecision()
    __recall = BinaryRecall()
    __accuracy = BinaryAccuracy()
   
    __train_loss_per_epoch, __train_acc_per_epoch, __train_precs_per_epoch, __train_rec_per_epoch = [], [], [], []
    __valid_loss_per_epoch, __valid_acc_per_epoch, __valid_precs_per_epoch, __valid_rec_per_epoch = [], [], [], []
    
    def __init__(self, model, dataloaders, loss_fn, optimizer, save_filename = None):
        self._set_device()
        
        self.__model = model.to(self.__device)
        self.__train = dataloaders['train']
        self.__valid = dataloaders['valid']
        self.__test = dataloaders['test']
        self.__loss_fn = loss_fn
        self.__optimizer = optimizer
        self.__save_filename = save_filename
        
        np.set_printoptions(formatter={'float_kind':"{:.6f}".format})
        
    def _set_device(self):
        if torch.cuda.is_available():
            self.__device = 'cuda'
    
    def _train(self):
        dataloader = self.__train
        
        train_loss, train_acc, train_precision, train_recall = 0, 0, 0, 0
        num_batches = len(dataloader)
        size = len(dataloader.dataset)
        
        self.__model.train()
        start = time.time()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.__device), y.to(self.__device)

            pred = self.__model(X)

            loss = self.__loss_fn(pred, y.float())
            acc = self.__accuracy(pred.cpu(), y.cpu())
            precision = self.__precision(pred.cpu(), y.cpu())
            recall = self.__recall(pred.cpu(), y.cpu())
            
            train_loss += loss.item()
            train_acc += acc
            train_precision += precision
            train_recall += recall
            

            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            if batch % 100 == 0:
                end = time.time()
                loss, current = loss.item(), batch * len(X)
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}] time: {end-start} acc: {acc} precision: {precision} recall: {recall}')
                start = time.time()

        train_loss /= num_batches
        train_acc /= num_batches
        train_precision /= num_batches
        train_recall /= num_batches

        self.__train_loss_per_epoch.append(train_loss)
        self.__train_acc_per_epoch.append(train_acc)
        self.__train_precs_per_epoch.append(train_precision)
        self.__train_rec_per_epoch.append(train_recall)

    def _train_autoencoder(self):
        dataloader = self.__train
        
        train_loss = 0
        num_batches = len(dataloader)
        size = len(dataloader.dataset)
        
        self.__model.train()
        start = time.time()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.__device), y.to(self.__device)

            pred = self.__model(X, y)
            loss = self.__loss_fn(pred, X)
            
            train_loss += loss.item()
            
            loss.backward()
            self.__optimizer.step()
            self.__optimizer.zero_grad()

            if batch % 100 == 0:
                end = time.time()
                loss, current = loss.item(), batch * len(X)
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}] time: {end-start}')
                start = time.time()

        train_loss /= num_batches

        self.__train_loss_per_epoch.append(train_loss)

    def _valid_autoencoder(self, test = False):
        dataloader = self.__valid
        print_keyword = 'Valid'
        
        if test:
            dataloader = self.__test
            print_keyword = 'Test'
        
        valid_loss = 0
        num_batches = len(dataloader)
        
        self.__model.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.__device), y.to(self.__device)
                pred = self.__model(X, y)
                valid_loss += self.__loss_fn(pred, X.float()).item()

        valid_loss /= num_batches

        print(f'{print_keyword} | Error: \n Avg loss: {valid_loss:>8f} \n')

        if not test:
            if valid_loss < self.__best_valid_loss and self.__save_filename:
                print(f'Saving model to file: {self.__save_filename}, {valid_loss} < {self.__best_valid_loss}')
                torch.save(self.__model.state_dict(), self.__save_filename)
                self.__best_valid_loss = valid_loss

            self.__valid_loss_per_epoch.append(valid_loss)

    def _valid(self, test = False):
        dataloader = self.__valid
        print_keyword = 'Valid'
        
        if test:
            dataloader = self.__test
            print_keyword = 'Test'
        
        valid_loss, valid_acc, valid_precision, valid_recall = 0, 0, 0, 0
        num_batches = len(dataloader)
        
        self.__model.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.__device), y.to(self.__device)
                pred = self.__model(X)
                valid_loss += self.__loss_fn(pred, y.float()).item()
                valid_acc += self.__accuracy(pred.cpu(), y.cpu())
                valid_precision += self.__precision(pred.cpu(), y.cpu())
                valid_recall += self.__recall(pred.cpu(), y.cpu())

        valid_loss /= num_batches
        valid_acc /= num_batches
        valid_precision /= num_batches
        valid_recall /= num_batches

        print(f'{print_keyword} | Error: \n Accuracy: {valid_acc:>8f}, Precision: {valid_precision:>8f}, Recall: {valid_recall:>8f}, Avg loss: {valid_loss:>8f} \n')

        if not test:
            if valid_loss < self.__best_valid_loss and self.__save_filename:
                print(f'Saving model to file: {self.__save_filename}')
                torch.save(self.__model.state_dict(), self.__save_filename)
                self.__best_valid_loss = valid_loss

            self.__valid_loss_per_epoch.append(valid_loss)
            self.__valid_acc_per_epoch.append(valid_acc)
            self.__valid_precs_per_epoch.append(valid_precision)
            self.__valid_rec_per_epoch.append(valid_recall)

    
    def run_classifier_training(self, epochs = 10):
        for t in range(epochs):
            print(f'Epoch {t+1}\n-------------------------------')
            self._train()
            self._valid()
        print('Done!')
        
        self._valid(test=True)

    def run_autoencoder_training(self, epochs = 10):
        for t in range(epochs):
            print(f'Epoch {t+1}\n-------------------------------')
            self._train_autoencoder()
            self._valid_autoencoder()
        print('Done!')
        
        self._valid_autoencoder(test=True)
        
        
    def load_weights(self, save_filename):
        self.__model.load_state_dict(torch.load(save_filename))
    
    def predict(self, x):
        with torch.no_grad():
            x = x.to(self.__device)
            pred = self.__model(x)

        return pred.cpu()
    
    def plot_metrics(self):
        pass
