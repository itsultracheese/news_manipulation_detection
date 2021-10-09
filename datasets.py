# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:04:02 2021

@author: Sergei
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

def extend(array, actual_len):
    pattern = np.zeros_like(array[0])
    for i in range(len(array)):
        l = len(array[i])
        for  j in range(actual_len - l - 1):
            array = np.concatenate([array,pattern.reshape(1,-1)], axis = 0)
    return array

def to_same_size(vectors):
        len_of_texts = [len(vectors[i]) for i in range(len(vectors))]
        actual_len = max([len(vectors[i]) for i in range(len(vectors))])
        print(actual_len)
        for i in range(len(vectors)):
            print(i)
            vectors[i] = extend(vectors[i], actual_len)
        return np.stack(vectors, axis = 0)

'''class lstm_dataset(torch.utils.data.Dataset):
    def __init__(self, vectors, targets):
        super().__init__()

        self.targets = to_same_size(targets, actual_len)
        self.vectors = to_same_size(vectors, actual_len)
    def __getitem__(self, i):
        return self.vectors[i], self.targets[i]
    def __len__(self):
        return len(self.vectors)'''
    
class lstm_dataset_simple(torch.utils.data.Dataset):
    def __init__(self, vectors, targets):
        super().__init__()
        self.targets = targets
        self.vectors = vectors
    def __getitem__(self, i):
        return self.vectors[i], self.targets[i]
    def __len__(self):
        return len(self.vectors)
        
def data2loader(vectors, targets):
    dset = lstm_dataset_simple(vectors, targets)
    lstmloader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=True)
    return lstmloader
    
        
        