#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
# import tensorflow as tf

def FedAvg(w, w_global, ratio=1, k=10, compression='Spar', scheduling='random', blocks=1):

    # w is list of objects. Every object contains two enteries
    # w[.] = [net.state_dict, sum(epoch_loss) / len(epoch_loss)]
    # copy.deepcopy(w[0]) = collections.OrderedDict

    if scheduling == "BN2":
        return bn2(w, w_global, k, ratio)

    ## Select Compression scheme
    if compression == 'None':
        return noCompression(w)
    elif compression == 'Spar':
        return randomSparcification(w, w_global, ratio/blocks)
    elif compression == 'Quant':
        return stochasticQuantization(w, ratio)
    elif compression == 'ML2':
        pass
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

def randomSparcification(w, w_global, ratio):
    ## Set random values to w_gloabal
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_diff = w[i][k] - w_global[k]
            w_diff_dropout = F.dropout(w_diff, p=ratio[i])
            w_avg[k] += w_diff_dropout + w_global[k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def stochasticQuantization(w, ratio, defualt_bit_depth=32):
    # https://github.com/scottjiao/Gradient-Compression-Methods/blob/master/utils.py
    w_avg = copy.deepcopy(w[0])
    arg = torch.device('cuda:{}'.format("0") if torch.cuda.is_available() else 'cpu')
    x_norm = torch.zeros(len(w)).to(arg)
    # Calculate norm beofre, so it's the same for all layers per user. 
    for k in w_avg.keys():
        for i in range(1, len(w)):
            x=w[i][k].float()
            x_norm[i] += torch.norm(x, p=float('inf'))

    for k in w_avg.keys():
        for i in range(1, len(w)):
            x=w[i][k].float()
            levels = round(defualt_bit_depth * (1 - ratio[i]))
            # x_norm = torch.norm(x, p=float('inf'))
            sgn_x=((x>0).float()-0.5)*2
            p=torch.div(torch.abs(x),x_norm[i])
            renormalize_p=torch.mul(p,levels)
            floor_p=torch.floor(renormalize_p)
            compare=torch.rand_like(floor_p)
            final_p=renormalize_p-floor_p
            margin=(compare < final_p).float()
            xi=(floor_p+margin)/levels
            w_avg[k] += x_norm[i]*sgn_x*xi
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def bn2(w, w_global, k_value, ratio):
    w_avg = copy.deepcopy(w[0])
    l2 = np.zeros(len(w))
    for k in w_avg.keys():
        for i in range(0, len(w)):
            l2[i] += torch.norm(w[i][k] - w_global[k], p=2)

    sorted = np.flip(np.argsort(l2))[0:k_value]
    print("idxs_users: " + str(sorted))
    print("compress_ratio: " + str(ratio[sorted]))
    print("Norm value: " + str(l2[sorted]))
    w_scheduled = [copy.deepcopy(w[i]) for i in sorted]
    return randomSparcification(w_scheduled, w_global, ratio)


## Old

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

