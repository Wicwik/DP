import pickle
import h5py

from torchvision import transforms

from utils.custom_dataset import CustomDataset

def load(data_path, targets_path):
    preds = None
    with open(targets_path,'rb') as f:
        preds = pickle.load(f)

    latents = None
    with h5py.File(data_path, 'r') as f:
        latents = f['z'][:]
        
    return latents, preds


def load_custom_dataset(data_path, targets_path, transform = None, target_transform = None):
    preds = None
    with open(targets_path,'rb') as f:
        preds = pickle.load(f)

    latents = None
    with h5py.File(data_path, 'r') as f:
        latents = f['z'][:]
        
    return CustomDataset(latents, preds, transform=transform, target_transform=target_transform)
