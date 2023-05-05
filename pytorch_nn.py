import torch
import time

import numpy as np

from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import MultilabelPrecision
from torchmetrics.classification import MultilabelRecall
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import MultilabelF1Score

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
    __f1 = BinaryF1Score()

    __experiment = None
    __epoch = None
   
    __loss_per_epoch, __acc_per_epoch, __precs_per_epoch, __rec_per_epoch, __f1_macro_per_epoch  = {}, {}, {}, {}, {}

    __loss_per_epoch['train'], __acc_per_epoch['train'], __precs_per_epoch['train'], __rec_per_epoch['train'], __f1_macro_per_epoch['train'] = [], [], [], [], []
    __loss_per_epoch['valid'], __acc_per_epoch['valid'], __precs_per_epoch['valid'], __rec_per_epoch['valid'], __f1_macro_per_epoch['valid'] = [], [], [], [], []
    
    def __init__(self, model, dataloaders, loss_fn, optimizer, save_filename = None, experiment = None, multilabel = True):
        self._set_device()
        
        self.__model = model.to(self.__device)
        self.__train = dataloaders['train']
        self.__valid = dataloaders['valid']
        self.__test = dataloaders['test']
        self.__loss_fn = loss_fn
        self.__optimizer = optimizer
        self.__save_filename = save_filename
        self.__experiment = experiment

        if model.n_classes > 1 and multilabel:
            self.__precision = MultilabelPrecision(model.n_classes)
            self.__recall = MultilabelRecall(model.n_classes)
            self.__accuracy = MultilabelAccuracy(model.n_classes)
            self.__f1 = MultilabelF1Score(model.n_classes)
        
        np.set_printoptions(formatter={'float_kind':"{:.6f}".format})
        
    def _set_device(self):
        if torch.cuda.is_available():
            self.__device = 'cuda'
        
    
    def _train(self):
        dataloader = self.__train
        
        train_loss, train_acc, train_precision, train_recall, train_f1_macro = 0, 0, 0, 0, 0
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
            f1_macro = self.__f1(pred.cpu(), y.cpu())

            if self.__experiment:
                metrics = {'train_loss': loss.item(), 'train_accuracy': acc, 'train_precision': precision, 'train_recall': recall, 'train_f1_macro': f1_macro}
                self.__experiment.log_metrics(metrics, step=batch)
            
            train_loss += loss.item()
            train_acc += acc
            train_precision += precision
            train_recall += recall
            train_f1_macro += f1_macro

            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            if batch % 100 == 0:
                end = time.time()
                loss, current = loss.item(), batch * len(X)
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}] time: {end-start} acc: {acc} precision: {precision} recall: {recall}, f1_macro: {f1_macro}')
                start = time.time()

        train_loss /= num_batches
        train_acc /= num_batches
        train_precision /= num_batches
        train_recall /= num_batches
        train_f1_macro /= num_batches

        self.__loss_per_epoch['train'].append(train_loss)
        self.__acc_per_epoch['train'].append(train_acc)
        self.__precs_per_epoch['train'].append(train_precision)
        self.__rec_per_epoch['train'].append(train_recall)
        self.__f1_macro_per_epoch['train'].append(f1_macro)

        if self.__experiment:
            metrics = {'avg_train_loss_per_epoch': train_loss, 
                       'avg_train_accuracy_per_epoch': train_acc, 
                       'avg_train_precision_per_epoch': train_precision, 
                       'avg_train_recall_per_epoch': train_recall,
                       'avg_train_f1_macro_per_epoch': train_f1_macro
                       }
            self.__experiment.log_metrics(metrics, step=self.__epoch)

    def _valid(self, test = False):
        dataloader = self.__valid
        print_keyword = 'Valid'
        
        if test:
            dataloader = self.__test
            print_keyword = 'Test'
        
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1_macro = 0, 0, 0, 0, 0
        num_batches = len(dataloader)
        
        self.__model.eval()
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.__device), y.to(self.__device)
                pred = self.__model(X)

                
                loss = self.__loss_fn(pred, y.float())
                acc = self.__accuracy(pred.cpu(), y.cpu())
                precision = self.__precision(pred.cpu(), y.cpu())
                recall = self.__recall(pred.cpu(), y.cpu())
                f1_macro = self.__f1(pred.cpu(), y.cpu())
                
                if self.__experiment:
                    metrics = {'valid_loss': loss, 'valid_accuracy': acc, 'valid_precision': precision, 'valid_recall': recall, 'valid_f1_macro': f1_macro}
                    self.__experiment.log_metrics(metrics, step=batch)

                valid_loss += loss.item()
                valid_acc += acc
                valid_precision += precision
                valid_recall += recall
                valid_f1_macro += f1_macro

        valid_loss /= num_batches
        valid_acc /= num_batches
        valid_precision /= num_batches
        valid_recall /= num_batches
        valid_f1_macro /= num_batches

        if self.__experiment:
            metrics = {f'avg_{print_keyword.lower()}_loss_per_epoch': valid_loss, 
                       f'avg_{print_keyword.lower()}_accuracy_per_epoch': valid_acc, 
                       f'avg_{print_keyword.lower()}_precision_per_epoch': valid_precision, 
                       f'avg_{print_keyword.lower()}_recall_per_epoch': valid_recall,
                       f'avg_{print_keyword.lower()}_f1_macro_per_epoch': valid_f1_macro
                       }
            if test:
                self.__experiment.log_metrics(metrics)
            else:
                self.__experiment.log_metrics(metrics, step=self.__epoch)
            

        print(f'{print_keyword} | Error: \n Accuracy: {valid_acc:>8f}, Precision: {valid_precision:>8f}, Recall: {valid_recall:>8f}, Avg loss: {valid_loss:>8f}, F1 macro: {valid_f1_macro:>8f} \n')

        if not test:
            if valid_loss < self.__best_valid_loss and self.__save_filename:
                print(f'Saving model to file: {self.__save_filename}')
                torch.save(self.__model.state_dict(), self.__save_filename)
                self.__best_valid_loss = valid_loss

            self.__loss_per_epoch['valid'].append(valid_loss)
            self.__acc_per_epoch['valid'].append(valid_acc)
            self.__precs_per_epoch['valid'].append(valid_precision)
            self.__rec_per_epoch['valid'].append(valid_recall)
            self.__f1_macro_per_epoch['valid'].append(valid_f1_macro)

    
    def run_classifier_training(self, epochs = 10):
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
