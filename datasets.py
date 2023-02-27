"""
File: datasets.py
------------------
This file holds various dataset and dataloading
functions. 
"""

import cloudpickle as cp
import torch
import pdb
from torch.utils.data import Dataset
import torchplate
from torchplate import utils as tp_utils
import requests 
from urllib.request import urlopen
import pandas as pd
import rsbox 
from rsbox import ml


class WaterbirdsDataset(Dataset):
    def __init__(self, folder_path, metadata_path, mode='train'):
        self.folder_path = folder_path
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(metadata_path)
        self.mode = mode
        self.split_map = {'train': 0, 'val': 1, 'test': 2}
        self.train_count = (self.metadata["split"] == self.split_map['train']).value_counts()[True]
        self.val_count = (self.metadata["split"] == self.split_map['val']).value_counts()[True]
        self.test_count = (self.metadata["split"] == self.split_map['test']).value_counts()[True]


        if mode == 'train':
            self.metadata = self.metadata[self.metadata['split'] == self.split_map['train']]
        elif mode == 'val':
            self.metadata = self.metadata[self.metadata['split'] == self.split_map['val']]
        elif mode == 'test':
            self.metadata = self.metadata[self.metadata['split'] == self.split_map['test']]
        else:
            raise ValueError('Invalid mode. Must be one of: train, val, test')

        self.data_distribution = self.metadata[['img_filename', 'y', 'place']].to_dict(orient='records')


        
    def __getitem__(self, index):

        img_file = self.data_distribution[index % len(self.data_distribution)]['img_filename']
        label = self.data_distribution[index % len(self.data_distribution)]['y']
        place = self.data_distribution[index % len(self.data_distribution)]['place']
        sample = ml.load_image(self.folder_path + '/' + img_file, resize=(256, 256), normalize=True)
        sample = torch.tensor(sample, dtype=torch.float)
        label = torch.tensor(label)
        return_dict = {'samples': sample, 'labels': label, 'places': place}
        return return_dict
        
    def __len__(self):
        return len(self.data_distribution)




# train_ds = WaterbirdsDataset(folder_path='../../../datasets/waterbird_data', metadata_path='../../../datasets/waterbird_data/metadata.csv', mode='train')

