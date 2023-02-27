"""
File: experiments.py
------------------
This file holds the experiments which are
subclasses of torchplate.experiment.Experiment. 
"""

import numpy as np
import torchplate
from torchplate import (
        experiment,
        utils
    )
from torchplate import metrics as tp_metrics
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datasets
import torchvision
import configs
import pdb


class BaseExp(experiment.Experiment):
    """
    Base experiment class. Simple supervised classification. 
    ---
    - Model: ResNet50 (imaenet pretrained)
    - Loss: CrossEntropyLoss
    - Optimizer: Adam 
    """
    
    def __init__(self, config=None):
        self.cfg = config
        if self.cfg is None:
            self.cfg = configs.BaseConfig()
        self.model = self.construct_model(pretrained=True, n_classes=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.train_ds = datasets.WaterbirdsDataset(folder_path=self.cfg.folder_path, metadata_path=self.cfg.metadata_path, mode='train')
        self.val_ds = datasets.WaterbirdsDataset(folder_path=self.cfg.folder_path, metadata_path=self.cfg.metadata_path, mode='val')
        self.test_ds = datasets.WaterbirdsDataset(folder_path=self.cfg.folder_path, metadata_path=self.cfg.metadata_path, mode='test')
        self.trainloader = torch.utils.data.DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True)
        self.valloader = torch.utils.data.DataLoader(self.val_ds, batch_size=self.cfg.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_ds, batch_size=1, shuffle=True)
        

        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = None,
            verbose = True
        )
    

    def construct_model(self, pretrained=True, n_classes=2):
        # source: https://github.com/kohpangwei/group_DRO/blob/master/run_expt.py 
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
        return model
    

    # provide this abstract method to calculate loss 
    def evaluate(self, batch):
        # batch will be of form 
        # {'samples': sample, 'labels': label, 'places': place}

        # get the samples and labels
        samples = batch['samples']
        labels = batch['labels']
        places = batch['places']

        logits = self.model(samples)
        loss_val = self.criterion(logits, labels)
        acc = tp_metrics.calculate_accuracy(logits, labels)

        metrics_dict = {'loss': loss_val, 'accuracy': acc}
        
        return metrics_dict

    
    def test(self):
        acc = tp_metrics.Accuracy()
        for batch in self.testloader:
            samples = batch['samples']
            labels = batch['labels']
            places = batch['places']
            logits = self.model(samples)
            acc.update(logits, labels)
        
        print(f'Accuracy: {acc.get()}')
    
    
            
