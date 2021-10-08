# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:33:32 2021

@author: Sergei
"""

from preprocessing import stem, create_dataset

path_train_articles = 'C:/sirius_october2021/datasets/train-articles/'
path_train_labels = 'C:/sirius_october2021/datasets/train-labels-task-flc-tc/'

texts, labels, article_names = create_dataset(path_train_articles, path_train_labels)

articles_stem, labels_stem = stem(texts, labels)
from Glove import text2vec, download

glove = download()
vectors = text2vec(glove,articles_stem[0])