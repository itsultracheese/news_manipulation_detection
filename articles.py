# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:32:08 2021

@author: Sergei
"""

import numpy as np
import pandas as pd
import os
 



path_train_articles = 'C:/sirius_october2021/datasets/train-articles/'
path_train_labels = 'C:/sirius_october2021/datasets/train-labels-task-flc-tc/'
articles = {}

art_files = os.listdir(path_train_articles)
for file in art_files:
    i = file[7:-4]
    with open(path_train_articles + file, 'r', encoding='utf-8') as fp:
        try:
            articles[i] = fp.read()
        except UnicodeDecodeError:
            print('Error ', i)

#for i in range(111111111,111111138):
 #   try:
  #      article = path_train_articles + f'article{i}.txt'
   #     with open(article, 'r') as fp:
    #        articles[i] = fp.read()
    #except:
     #   continue
    
    
teg_files = os.listdir(path_train_labels)
tegs = {}
for file in teg_files:
    i = file[7:16]
    with open(path_train_labels + file, 'r') as fp:
        tegs[i] = fp.read()
        
def take_manipulation(teg):
    return teg.split()[1]

def set_borders(num, articles, tegs):
    art = articles[num]
    t = tegs[str(num)]
    b = []
    mans = []
    for s in t.split('\n'):
        if s != '':
            b.append([s.split()[2],s.split()[3]])
            mans.append(take_manipulation(s))
    art = art
    i = 0
    for B in b:
        a =int( B[0])
        b_ = int(B[1])
        art = art[:a] + '     !!!{' + art[a:b_] + '}!!!   (MANIPULATION: ' + mans[i] + ')' + art[b_:]
        i = i+1
        
    return art
        
    

d = {}

for k in tegs.keys():
    for s in tegs[k].split('\n'):
        if s != '':
            try:
                d[take_manipulation(s)] += 1
            except KeyError:
                d[take_manipulation(s)] = 1
    
    
    

d_1 = {d[k] : k for k in d.keys()}
sorted_k = sorted(list(d_1.keys()))

c = 0
for t in tegs.keys():
    if 'Loaded_Language' in [take_manipulation(z) for z in tegs[t].split('\n')[:-1]]:
        c+=1