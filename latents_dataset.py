import pickle
import h5py
import numpy as np

from utils.custom_dataset import CustomDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class LatentsDataset:
    __targets_path = None
    __data_path = None
    scaler = None

    def __init__(self, data_path, targets_path):
        self.__data_path = data_path
        self.__targets_path = targets_path

    def load(self):
        preds = None
        with open(self.__targets_path,'rb') as f:
            preds = pickle.load(f)

        latents = None
        with h5py.File(self.__data_path, 'r') as f:
            latents = f['z'][:]

        return latents, preds

    def load_custom_dataset(self, transform = None, target_transform = None, balanced_classes = False, minmax_norm = False):
        preds = None
        with open(self.__targets_path,'rb') as f:
            preds = pickle.load(f)

        latents = None
        with h5py.File(self.__data_path, 'r') as f:
            latents = f['z'][:]

        if minmax_norm:
            self.scaler = MinMaxScaler()
            latents = self.scaler.fit_transform(latents)

        preds = np.round(preds)
            
        return CustomDataset(latents, preds, transform=transform, target_transform=target_transform)

    def load_autoencoder_dataset(self, transform = None, minmax_norm = False):
        latents = None
        with h5py.File(self.__data_path, 'r') as f:
            latents = f['z'][:]

        if minmax_norm:
            self.scaler = MinMaxScaler()
            latents = self.scaler.fit_transform(latents)

        return CustomDataset(latents, latents, transform=transform, target_transform=transform)
