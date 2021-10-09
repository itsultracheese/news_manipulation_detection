# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:40:52 2021

@author: Sergei
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl

class LSTM_tagger(nn.Module):
    def __init__(self, dimension=300, hidden_size=20, num_layers=1, dropout_rate=0.01, activation='relu'):
        super().__init__()
        self.dim = dimension
        self.lstm = nn.LSTM(input_size=dimension, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            dropout=dropout_rate)
        
        self.activation = nn.ELU() #HARDCODE
        
        self.FF = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=2), 
                                self.activation, 
                                nn.Softmax(dim=-1))
        
    def forward(self, sequence : torch.Tensor):
        X = sequence
        lstm_output = self.lstm(X)[0]
        
        probabilities = self.FF(lstm_output)
        output = torch.argmax(probabilities, dim = -1)
        return probabilities, output

    
class Shell(pl.LightningModule):
    def __init__(self, model, CONFIG, device='cpu'):
        super().__init__()
        self.logs = {'loss':[]}
        self.model = model
        self.lr = CONFIG['learning_rate']
        self.optim = CONFIG['optimizer']
        self.classweights = CONFIG['crossentropy_weights']
        self.print_every_n = CONFIG['print_every_n']
        self.lossfunc = nn.CrossEntropyLoss(weight = self.classweights)
        
        
    def configure_optimizers(self):
        if self.optim == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {'optimizer' : optimizer}
    
    def step(self, batch, step, mode='train', to_print = False):
        if mode == 'train':
            self.model.train()
        if mode == 'test':
            self.model.eval()
        
        X, tags = batch
        X = X.float()
        tags = tags.long()
        probabilities, output = self.model(X)
        
        loss = self.lossfunc(probabilities.view(-1,probabilities.shape[-1]), tags.view(-1))
        self.logs['loss'].append(loss.item())
        if to_print:
            print(f'step {step} : {mode} loss = {loss.item()}')
        if mode == 'train':
            return loss
    
    def training_step(self, batch, step):
        to_print = True if step % self.print_every_n == 0 else False
        loss = self.step(batch, step, mode='train', to_print=to_print)
        return loss
    
    def validation_step(self, batch, step):
        to_print = True if step % self.print_every_n == 0 else False
        self.step(batch, step, mode='val', to_print=to_print)