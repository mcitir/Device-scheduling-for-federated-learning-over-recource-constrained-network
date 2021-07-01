#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np

def FedAvg(w, w_global, scheduling='L2'):

    # w is list of objects. Every object contains two enteries
    # w[.] = [net.state_dict, sum(epoch_loss) / len(epoch_loss)]
    # copy.deepcopy(w[0]) = collections.OrderedDict

    ## Select scheduling scheme
    if scheduling == 'None':
        w_avg = noScheduling(w)
    elif scheduling == 'L2':
        w_avg = simpleL2(w, w_global)
    else:
        print("No scheduling selected")
        w_avg = noScheduling(w)
    return w_avg

def noScheduling(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def simpleL2(w, w_global):
    # Very simple and naive scheduling. This scheduling calculates the
    # the norms of all the layers, for all the users, and only avarages
    # the ones with the biggest norms. 

    ratio = 0.2 # The ratio of updates used
    updatesReduction = len(w) - round(len(w)*ratio)
    w_avg = copy.deepcopy(w[0])
    w_g = copy.deepcopy(w_global)
    l2 = np.zeros(len(w))
    # Loop users
    for i in range(0, len(w)):
        for k in w_avg.keys():
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
        