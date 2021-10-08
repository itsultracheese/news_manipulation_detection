# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:35:21 2021

@author: Sergei
"""
from preprocessing import stem_, stem, create_dataset
import numpy as np
PATH = 'C:\sirius_october2021\glove\multilingual_embeddings'
LANG = '.en'

def download(lan=LANG):
    glove = {}
    with open(PATH + lan, 'r', encoding='utf-8') as fp:
        for i,line in enumerate(fp):
            if ~i%10000:
                print(f'{i} words are downloaded')
            vec = line.split()
            word = vec[0]
            if lan == '.en':
                try:
                    word = stem_([[word]])[0][0]
                except IndexError:
                    word = word
            vector = list(map(lambda x: float(x), vec[1:]))
            glove[word] = vector
    return glove

def get_word(w, lan=LANG):
    with open(PATH + lan, 'r', encoding='utf-8') as fp:
        for i,line in enumerate(fp):
            vec = line.split()
            word = vec[0]
            if w == word:
                vector = list(map(lambda x: float(x), vec[1:]))
                return vector

def text2vec(glove : dict, text_tokenized: list()):
    vectors = [glove[word] if word in glove.keys() else np.zeros((300)) for word in text_tokenized]
    vectors = np.concatenate(vectors, axis=-1)
    return vectors
    
def path2dataset(path_train_articles, path_train_labels):
    texts, labels, article_names = create_dataset(path_train_articles, path_train_labels)

    articles_stem, labels_stem = stem(texts, labels)
    targets = [np.array(t) for t in labels_stem]

    glove = download()
    V = []
    for text in articles_stem:
        vectors = text2vec(glove,text)
        V.append(vectors)
    return V, targets
    