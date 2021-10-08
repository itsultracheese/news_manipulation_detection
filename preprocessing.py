# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 15:22:38 2021

@author: Sergei
"""

import pandas as pd
import os

import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import glob
import os
import codecs
import numpy as np


def read_articles_from_file_list(folder_name, file_pattern="*.txt"):
    '''
    Read articles from files matching patterns <file_pattern> from  
    the directory <folder_name>. 
    The content of the article is saved in the dictionary whose key
    is the id of the article (extracted from the file name).
    Each element of <sentence_list> is one line of the article.
    '''
    file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles = {}
    article_id_list, sentence_id_list, sentence_list = ([], [], [])
    for filename in sorted(file_list):
        article_id = os.path.basename(filename).split('.')[0][7:]
        with codecs.open(filename, 'r', encoding='utf8') as f:
            articles[article_id] = f.read()
    return articles

def read_predictions_from_file(filename):
    '''
    Reader for the gold file and the template output file. 
    Return values are four arrays with article ids, labels 
    (or ? in the case of a template file), begin of a fragment, 
    end of a fragment. 
    '''
    articles_id, span_starts, span_ends, gold_labels = ([], [], [], [])
    with open(filename, 'r') as f:
        for row in f.readlines():
            article_id, gold_label, span_start, span_end = row.rstrip().split('\t')
            articles_id.append(article_id)
            gold_labels.append((gold_label, int(span_start), int(span_end)))
    return articles_id, gold_labels

def label(text, gt_labels):
    tokens = []
    labels = []
    special_symbols = """!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~ \n\t\'\\"""
    word = ''
    inside = False
    word_start = 0
    for i in range(len(text)):
        if text[i] in special_symbols:
            if len(word) > 1:
                tokens.append(word)
                word = ''
                if inside:
                    if gt_labels[0][1] == word_start:
                        labels.append(2)
                    else:
                        labels.append(1)
                else:
                    labels.append(0)
        else:
            if len(word) == 0:
                word_start = i
            word += text[i]
        if len(gt_labels) > 0:
            if i == gt_labels[0][1]:
                inside = True
            elif i == gt_labels[0][2] + 1:
                inside = False
                gt_labels.pop(0)
    return tokens, labels
    

def create_dataset(path_to_articles, path_to_labels):
    '''
    Creates the dataset from the files contained in 'datasets/train-articles/' folder
    
    texts : list, each represents one article and contains
    '''
    texts = []
    labels = []
    articles = read_articles_from_file_list(path_to_articles)
    article_names = list(articles.keys())
    prefix_lbl = path_to_labels + '/article'
    postfix_lbl = '.task-flc-tc.labels'
    for name in article_names:
        articles_id, gold_labels = read_predictions_from_file(prefix_lbl + name + postfix_lbl)
        gt_labels = []
        for i in range(len(gold_labels)):
            if gold_labels[i][0] == 'Loaded_Language':
                gt_labels.append(gold_labels[i])
        gt_labels.sort(key=lambda x: x[1])
        tokens, lbls = label(articles[name], gt_labels)
        texts.append(tokens)
        labels.append(lbls)
    
    return texts, labels, article_names

porter = PorterStemmer()
stopwords = set(stopwords.words('english'))

def stem_(articles):
    articles_stem = []
    sents = 0
    for article in articles:
        article_stem = []
        for word in article:
            if word.lower() not in stopwords:
                word_new = porter.stem(word.lower())
                article_stem.append(word_new)
        articles_stem.append(article_stem)
        sents += 1 
    return articles_stem

def stem(articles, labels):
    labels_stem = []
    articles_stem = []
    sents = 0
    for article in articles:
        article_stem = []
        label_stem = []
        for word in article:
            if word.lower() not in stopwords:
                cnt = article.index(word)#.lower())
                label_stem.append(labels[sents][cnt]) 
                word_new = porter.stem(word.lower())
                article_stem.append(word_new)
        labels_stem.append(label_stem)
        articles_stem.append(article_stem)
        sents += 1 
    return articles_stem, labels_stem

def clean(text):
    clean_text = ''
    text = text.replace('!',' ! ').replace('.', ' ').replace('?', ' ').replace('<',' ').replace('>', ' ').replace('/',' ')
    for s in text:
        if "а" <= s.lower() <= "я" or s == ' ':
            clean_text+=s
    return clean_text

#def text_to_vectors