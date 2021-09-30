#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def FedAvg(w, w_global, ratio=1, compression='None'):

    # w is list of objects. Every object contains two enteries
    # w[.] = [net.state_dict, sum(epoch_loss) / len(epoch_loss)]
    # copy.deepcopy(w[0]) = collections.OrderedDict

    ## Select scheduling scheme
    if compression == 'None':
        w_avg = noCompression(w)
    elif compression == 'random':
        w_avg = randomCompression(w, w_global, ratio)
    elif compression == 'ML2':
        multiLayerL2(w, w_global)
    else:
        print("No Compression")
        w_avg = noCompression(w)
    return w_avg

def noCompression(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def randomCompression(w, w_global, ratio):
    ## Set random values to w_gloabal
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_diff = w[i][k] - w_global[k]
            w_diff_dropout = F.dropout(w_diff, p=ratio[i])
            w_avg[k] += w_diff_dropout + w_global[k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg








#### OLD

def simpleL2(w, w_global):
    # Very simple and naive scheduling. This scheduling calculates the
    # the norms of all the layers, for all the users, and only avarages
    # the ones with the biggest norms. 

    ratio = 0.2 # The ratio of updates used
    updatesReduction = len(w) - round(len(w)*ratio)
    w_g = copy.deepcopy(w_global)
    l2 = np.zeros(len(w))
    # Loop users
    for i in range(0, len(w)):
        for k in w_g.keys():
            # The difference between local and global is the interesting part
            l2[i] += torch.norm(w[i][k] - w_g[k])
    sorted = np.argsort(l2)[updatesReduction:]
    print(  str(len(w)) + "   " + str(len(sorted)))
    print(str(sorted))
    print(str(l2[sorted]))

    w_avg = copy.deepcopy(w[sorted[0]])
    for k in w_avg.keys():
        for i in sorted[1:]:
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(sorted))
    return w_avg
        




# Does not work
def multiLayerL2(w, w_global):
    ratio = 0.2 # The ratio of updates used
    updatesReduction = len(w) - round(len(w)*ratio)
    w_g = copy.deepcopy(w_global)
    l2 = np.zeros((len(w), len(w_g.keys())))
    for i in range(0, len(w)):
        for k in w_g.keys():
            # The difference between local and global is the interesting part
            l2[i][k] = torch.norm(w[i][k] - w_g[k])
    sorted = np.zeros(len(w)-updatesReduction, len(w_g.keys() ))
    for k in w_g.keys():
        sorted[:][k] = np.argsort(l2[:][k])[updatesReduction:]
    
    w_avg = copy.deepcopy(w[sorted[0][:]])
    for k in w_avg.keys():
        for i in sorted[1:][k]:
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(sorted))
    return w_avg

