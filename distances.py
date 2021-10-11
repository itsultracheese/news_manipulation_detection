# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 09:48:47 2021
Набор вспомогательных функций для MyExperiment
@author: Sergei
"""


from scipy.stats import entropy
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import re                    
import numpy as np
from numba import guvectorize
from ewm import ewma
import math


#Normalizing function
def NORM(to_norm):
    return np.sqrt(to_norm)/(np.sqrt(to_norm) + 1)


#def hellinger(p, q):
    #return math.sqrt(sum([ (math.sqrt(p_i) - math.sqrt(q_i))**2 for p_i, q_i in zip(p, q) ]) / 2)
 #   return 
def hellinger(p, q):
    return np.sqrt(1/2) * np.sqrt(  np.sum((np.sqrt(p) - np.sqrt(q))**2)  )
#The mean of cosine distances between news in chain and argument new
def Cosine_measure_to_chain_av(chain, new):
    Sum = np.array([cosine(np.array(c), np.array(new)) for c in chain ]).sum()
    Sum/= len(chain)
    #print(Sum)
    return Sum

#The weighted mean of cosine distances between news in chain and argument new
def Cosine_measure_to_chain_w_av(chain, new):
    distances = [cosine(np.array(c), new) for c in chain[-4:]]
    m = ewma(np.array(distances))
    return m[-1]
    
#The EWM of cosine distances between news in chain and argument new
def Cosine_measure_to_chain_asMin(chain, new):
        distances = np.array([cosine(np.array(c), new) for c in chain ])
        ans = distances.max()
        return ans
#The distance from new and chain as cosine measure between new and the last 
def Cosine_measure_to_chain_last(chain, new):
    distance = cosine(chain[-1], new)
    return distance


                            #Euclidian Measure
    
#The mean of Euclidian distances between news in chain and argument new
def Euclidian_measure_to_chain_av(chain, new):
    chain = np.array(chain)
    new= np.array(new)
    chain = chain.reshape(chain.shape[0],-1)
    arr = np.linalg.norm(chain - new,axis = 1)
    h = arr.mean()
    normalized = NORM(h)
    return normalized

#The weighted mean of cosine distances between news in chain and argument new
def Euclidian_measure_to_chain_w_av(chain, new):
    distances = [np.linalg.norm(np.array(c) - new) for c in chain ]
    m = ewma(np.array(distances))
    normalized = NORM(m[-1])
    return normalized
    
#The EWM of cosine distances between news in chain and argument new
def Euclidian_measure_to_chain_asMin(chain, new):
        distances = np.array([np.linalg.norm(np.array(c) - new) for c in chain ])
        ans = distances.max()
        normalized = NORM(ans)
        return normalized
    
#The distance from new and chain as cosine measure between new and the last 
def Euclidian_measure_to_chain_last(chain, new):
    distance = np.linalg.norm(chain[-1] - new)
    normalized = NORM(distance)
    return normalized



                            #Manhattan Measure
#The mean of Euclidian distances between news in chain and argument new
def Manhattan_measure_to_chain_av(chain, new):
    chain = np.array(chain)
    new= np.array(new)
    chain = chain.reshape(chain.shape[0],-1)
    arr = np.sum(np.abs(chain - new),axis = 1)
    h = arr.mean()
    normalized = NORM(h)
    return normalized


#The weighted mean of cosine distances between news in chain and argument new
def Manhattan_measure_to_chain_w_av(chain, new):
    chain = np.array(chain)
    new= np.array(new)
    chain = chain.reshape(chain.shape[0],-1)
    arr = np.sum(np.abs(chain - new),axis = 1)
    m = ewma(np.array(arr))
    normalized = NORM(m[-1])
    return normalized
    
    
    '''
    distances = pd.Series([np.sum(np.abs(np.array(c) - new)) for c in chain ])
    mean = distances.ewm(3, min_periods = 0).mean()
    return mean.to_numpy()[-1]'''
    
#The EWM of cosine distances between news in chain and argument new
def Manhattan_measure_to_chain_asMin(chain, new):
        distances = np.array([np.sum(np.abs(np.array(c) - new)) for c in chain ])
        ans = distances.max()
        normalize = NORM(ans)
        return normalize
#The distance from new and chain as cosine measure between new and the last 
def Manhattan_measure_to_chain_last(chain, new):
    distance = np.sum(np.abs(np.array(chain[-1]) - new))
    normalize = NORM(distance)
    return normalize



    
                            #KL-Divergence 
#The mean of KL-Divergence  between news in chain and argument new
def KL_measure_to_chain_av(chain, new):
    chain = chain.reshape(chain.shape[0],-1)
    Sum = np.array([entropy(np.array(c), new) for c in chain ]).sum()
    Sum/= len(chain)
    normalize = NORM(Sum)
    return normalize

#The weighted mean of KL-Divergence  between news in chain and argument new
def KL_measure_to_chain_w_av(chain, new):
    chain = chain.reshape(chain.shape[0],-1)
    distances = [entropy(np.array(c) ,new) for c in chain ]
    m = ewma(np.array(distances))
    normalized = NORM(m[-1])
    return normalized
    
#The EWM of KL-Divergence  between news in chain and argument new
def KL_measure_to_chain_asMin(chain, new):
        chain = chain.reshape(chain.shape[0],-1)
        distances = np.array([entropy(np.array(c), new) for c in chain ])
        ans = distances.max()
        normalized = NORM(ans)
        return normalized
#The distance from new and chain as KL-Divergence  between new and the last 
def KL_measure_to_chain_last(chain, new):
    chain = chain.reshape(chain.shape[0],-1)
    distance = entropy(chain[-1],new)
    normalized = NORM(distance)
    return normalized




#                       Hellinger distance
def Hel_measure_to_chain_av(chain, new):
    chain = chain.reshape(chain.shape[0],-1)
    Sum = np.array([hellinger(np.array(c), new) for c in chain ]).sum()
    Sum/= len(chain)
    normalize = NORM(Sum)
    return normalize

#The weighted mean of KL-Divergence  between news in chain and argument new
def Hel_measure_to_chain_w_av(chain, new):
    chain = chain.reshape(chain.shape[0],-1)
    distances = [hellinger(np.array(c) ,new) for c in chain ]
    m = ewma(np.array(distances))
    normalized = NORM(m[-1])
    return normalized
    
#The EWM of KL-Divergence  between news in chain and argument new
def Hel_measure_to_chain_asMin(chain, new):
        chain = chain.reshape(chain.shape[0],-1)
        distances = np.array([hellinger(np.array(c), new) for c in chain ])
        ans = distances.max()
        normalized = NORM(ans)
        return normalized
#The distance from new and chain as KL-Divergence  between new and the last 
def Hel_measure_to_chain_last(chain, new):
    chain = chain.reshape(chain.shape[0],-1)
    distance = hellinger(chain[-1],new)
    normalized = NORM(distance)
    return normalized


@guvectorize(['void(float64[:], intp[:], float64[:])'],
             '(n),()->(n)')
def move_mean(a, window_arr, out):
    window_width = window_arr[0]
    asum = 0.0
    count = 0
    for i in range(window_width):
        asum += a[i]
        count += 1
        out[i] = asum / count
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / count

#@guvectorize(['void(float64[:], intp[:], float64[:])'],
 #            '(n),()->(n)')
def move_weighted_mean_linear(a, window_arr, out):
    window_width = window_arr[0]
    weights = (1/2)**np.arange(window_width)
    vec = a[-window_width:]
    return np.dot(vec,weights)