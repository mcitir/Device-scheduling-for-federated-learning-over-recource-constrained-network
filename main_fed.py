#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from utils.scheduling import Scheduler

import signal
import sys
import atexit

from utils.custom_utils import can_complete_task, apply_preemption
import random

# Global variables
SEED_VALUE = 2023
MIN_RATIO = 0.6
MAX_RATIO = 1.2
EPOCH_TIME = 600 # Deadline Constraint, 10 minutes in seconds

# Predefined delays for unique process
MAX_PREEMPTION_DELAY = 60
PREEMPTION_PROBABILITY = 0.2 # 20% probability for applying preemption to a user
preempted_users = None

USER_RESELECTION_DELAY = 10

# Calculate dataset size for each user by using total dataset size and number of users
UNIFORM_DATASET_SIZE = None # This will be calculated in the main loop; it defines the size of the dataset for each user in the case of uniform distribution
MAX_COMPUTATION = 0.08
FLUCTUATION_COMPUTATION = 30


# Usecase specific variables
UNIFORM_DATASIZE_DISTRIBUTION = True # If it is True, data is distributed uniformly among users

# Interruption parameters
LAMBDA_I = 0.004 # Avarage rate at which interruptions occur per unit time (per second), 0.004 means 4 interruption every 1000 seconds
MU_K = 30 # Execution rate (interruption duration) in seconds


##################################################
######## SETUP ###################################
if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # Finding the size of the dataset for each user by using the total dataset size and number of users
    if UNIFORM_DATASIZE_DISTRIBUTION:
        UNIFORM_DATASET_SIZE = len(dataset_train) / args.num_users
    else:
        raise ValueError("Error in dataset size distribution") # This is not implemented yet

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    acc = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # scheduling
    k = max(int(args.frac * args.num_users), 1)
    selectedUsers = np.zeros(args.num_users)
    scheduler = Scheduler(args.num_users, dict_users, dataset_train, args.snr, args.sched, args.comp,args.model,args.dataset)

    # This provides to generate a summary of the data when the program is interrupted
    def signal_handler(signum, frame):
        scheduler.logger.generate_summary()
        sys.exit(0)

    # Register the signal handler for interruption (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Register the method to be called upon normal termination
    atexit.register(scheduler.logger.generate_summary)

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]    


    # Computational capacities for each user 
    maximum_computation_capabilities =[MAX_COMPUTATION for _ in range(args.num_users)]
    fluction_computation_capabilities = [FLUCTUATION_COMPUTATION for _ in range(args.num_users)]
    

    ##################################################
    ##### MAIN LOOP ##################################
    for iter in range(args.epochs):
        # Print concept name
        print("Concept: " + args.concept)
        # Print current round
        print("Round: " + str(iter))
        
        # Reset available time for each user
        available_time = EPOCH_TIME

        loss_locals = []
        if not args.all_clients:
            w_locals = []


        ##################################################
        ##### UPLINk SCHEDULING ##########################
        #k = max(int(args.frac * args.num_users), 1) # Numer of users

        #idxs_users = np.random.choice(range(args.num_users), m, replace=False) # Select at random
        #idxs_users = userSelection(m, dict_users, dataset_train, selectedUsers, True)


        ## Here goes Hiroki's algorithm
            # Sangyoung writes a function here that models training time according to data size
            #  Hiroki can use this inside the algorithm
            # list of users & dataset sizes

            # probabilities.. regarding when actually the training finishes

            # list of users which completed training within time & dataset sizes
        
        # User selection
        idxs_users, compress_ratio = scheduler.newUsers(k, iter) # Iter is the current round number, used for logging
        # Track selected users
        selected_idxs = set(idxs_users)

        ## Modeling of the preemption
        # Sangyoung writes a function here that models preemption
        # idPreempted, timePreempted = preemption(idxs_users)

        # Apply preemption
        # Return list of tuples (user_id, preemption_time)
        # Num of users, probability of preemption, max preemption time (seconds)
        preempted_users = apply_preemption(range(args.num_users), PREEMPTION_PROBABILITY, MAX_PREEMPTION_DELAY)
        
        # Track users that couldn't complete the task
        idx_to_remove = []

        selectedUsers[idxs_users] += 1
        
        if args.sched != "BN2":
            print("idxs_users: " + str(idxs_users))
            print("compress_ratio: " + str(compress_ratio))


        for idx in idxs_users:
            
            # Check if user can complete task
            if not can_complete_task(fluction_computation_capabilities[idx], 
                                     maximum_computation_capabilities[idx], 
                                     EPOCH_TIME, UNIFORM_DATASET_SIZE, LAMBDA_I, 
                                     MU_K):
            
                print(f"User {idx} couldn't complete the task due to insufficient capacity.")
                idx_to_remove.append(idx)
            
        # Assign new user for each unsuccessful users
        for removed_idx in idx_to_remove:
            if args.concept == "dynamic":
                avaliable_users = set(range(args.num_users)) - selected_idxs
                last_tested_user = removed_idx

                while avaliable_users:

                    # Here, if the selected user can't complete the task, then we need to continue to select a new user.
                    # So, we need to continue reducing the available time for this user.
                    # Calculate available time for the user.
                    available_time -= USER_RESELECTION_DELAY
                    print(f"User {last_tested_user} couldn't be selected. Available time is now {available_time} seconds.")
                    
                    # !!! Here, user selection is done from the user pool, so there is no need to use the scheduler function
                    # Randomly pick a user except the ones who are already selected
                    new_idx = np.random.choice(list(avaliable_users))
                    last_tested_user = new_idx
                    avaliable_users.remove(new_idx)
                    
                    # Check if the new user can complete the task
                    if can_complete_task(fluction_computation_capabilities[last_tested_user],
                                            maximum_computation_capabilities[last_tested_user],
                                            available_time,
                                            UNIFORM_DATASET_SIZE, LAMBDA_I, MU_K):
                    # Add the newly chosen user to the set of selected users
                        selected_idxs.add(last_tested_user)
                        
                    
                        # Replace the user who couldn't complete the task (denoted by 'removed_idx') 
                        # with new candidate users
                        idxs_users = [last_tested_user if x == removed_idx else x for x in idxs_users]
                        print(f"User {last_tested_user} was randomly chosen to replace user {removed_idx}.")
                        break
                    else:
                        pass #print(f"User {new_idx} couldn't be replaced because it also can't complete the task.")
            else:
                # If the concept is not dynamic, then we just remove the user
                idxs_users.remove(removed_idx)

        # All users in idxs_users can complete the task
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        # added w_global to parameters
        w_glob = FedAvg(w_locals, w_glob, compress_ratio, k, args.comp, args.sched, args.blocks)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # Added accurency convergence
        acc_avg, _ = test_img(net_glob, dataset_test, args)
        acc.append(acc_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}_snr{}_comp{}_scheduling{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.snr, args.comp, args.sched))

    # save result
    np.save('./save/loss_fed_{}_{}_{}_C{}_iid{}_snr{}_comp{}_scheduling{}.npy'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.snr, args.comp, args.sched), loss_train)
    np.save('./save/acc_fed_{}_{}_{}_C{}_iid{}_snr{}_comp{}_scheduling{}.npy'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.snr, args.comp, args.sched), acc)
    if args.sched != 'BN2':
        np.save('./save/ni_fed_{}_{}_{}_C{}_iid{}_snr{}_comp{}_scheduling{}.npy'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.snr, args.comp, args.sched), selectedUsers)

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
