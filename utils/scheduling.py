import numpy as np
from numpy.random import standard_normal
from numpy import abs, log2, zeros
import csv
import os
from datetime import datetime
import torch
import pandas as pd

### Scheduler Class
class Scheduler:
    def __init__(self, m, dict_users, dataset_train, SNR, mode='old', comp='None', model_name='Unknown', dataset='Unknown'):
        self.rates = Rates(m, SNR)
        self.mode = mode
        self.picks = zeros(m)
        self.sinceLastPick = zeros(m)
        self.comp = comp
        self.dict_users = dict_users
        self.dataset_train = dataset_train
        self.model_name = model_name
        self.dataset_name = dataset
        self.logger = DataLogger("", self.model_name, self.dataset_name, self.mode) # Data Logger for logging data size and capacity

    def newUsers(self, k, iter): # iter is the current round number, used for logging
        self.rates.update()
        users = pick(self.rates, k, self.sinceLastPick, self.mode, self.dict_users, self.dataset_train)
        self.rates.ni[users] += 1
        self.picks[users] += 1
        self.sinceLastPick += 1
        self.sinceLastPick[users] = 0
        ratio = compressRatio(self.rates, users)

        # Log data size and capacity for each user
        for user in users:
            capacity = self.rates.getCapacity()[user] # Get the capacity for the user
            self.logger.log_data(round_num=iter, user=user, dataset=self.dataset_train, dict_users=self.dict_users, capacity=capacity)

     

        return users, ratio

def pick(rates, k, sinceLastPick, mode, dict, dataset):
    if mode == 'RS':
        return np.random.choice(range(rates.nbr_users,), k, replace=False)
    elif mode == 'BC':
        return capacityScheduling(rates, k)
    elif mode == 'BN2':
        return np.random.choice(range(rates.nbr_users,), rates.nbr_users, replace=False)
    elif ((mode == 'G1') or (mode =='G1-M')):
        return g1(rates, dict, dataset, k, mode)
    elif mode == 'old':
        return oldUsers(rates, sinceLastPick, k)

def oldUsers(rates, sinceLastPick, k):
    sortedIndex = np.argsort(sinceLastPick)
    return np.flip(sortedIndex)[0:k]

def compressRatio(rates, choosen, bit_depth=33):
    net_size=1000 # temp
    #net_size = rates.network_size

    dataPerUser = rates.bitsPerBlock[choosen]
    compresPerUser = dataPerUser / (net_size * bit_depth)
    compresPerUser[compresPerUser > 1] = 1
    return (1 - compresPerUser)

class Rates:
    def __init__(self, nbr_users, SNR=30):
        ## OFDM Constants
        self.Ts = (1.0/14) * 10**(-3)
        Tu = (1.0/15) * 10**(-3)
        Tcp = (1.0/(14*15)) * 10**(-3)
        Bs = 15*10**3
        self.Bc = 210*10**3
        N_smooth = 14
        T_slot = 2*10**(-3)
        N_slot = 28
        self.coherenceTime = self.Ts * N_slot

        ## Usefull stuff
        self.network_size=10**3         #self.network_size=10**3
        self.nbr_users = nbr_users
        self.samples = N_smooth*N_slot
        self.sampleSize = self.Ts * Bs # Maybe Tu?
        self.ni = np.zeros(nbr_users)

        ## Channel
        self.SNR = SNR
        self.update()
    
    def update(self):
        """
        Only change small scale fading, then update values. 
        """
        h = (standard_normal(self.nbr_users) + 1j * standard_normal(self.nbr_users)) * 1 * 0.5
        self.capacity = log2(1+pow(abs(h),2) * pow(self.SNR/10, 10))
        self.bitsPerSymbol = self.capacity*self.sampleSize  
        self.bitsPerBlock = self.capacity * self.Bc * self.coherenceTime

    def getBitsPerBlock(self):
        return self.bitsPerBlock

    def getCapacity(self):
        return self.capacity


# Scheduling Schemes

def capacityScheduling(rates, k):
    ## Return the k users with the highest capacity
    capacity = rates.bitsPerSymbol
    sorted = np.argsort(capacity)
    return np.flip(sorted)[0:k]

def search(rates, distrobution, m, iterations, mode):
    """
    Given the distrobution matrix, this function searches for a 
    linear combination of m colums that are as evenly distubuted 
    as possible. No analytical solution exists, and to go through 
    all possible solutions would take to much time, so the 
    function guesses linear combinations and picks the best of all
    guesses based on the variance of the guessed linear combination. 

    The function also prioritize users which has been picked less 
    seldome, by bumping their probability to be guessed.

    distrobution: Matrix with the distrobution of classes for all users
    m: number of users to pick
    iterations: number of guesses
    selectedUsers: How users has previously been selected. 
    """
    var_min = float('inf')
    bucket_min = np.zeros(distrobution.shape[0])

    # Increase the probability to pick users selected more seldome
    if mode == 'G1-M':
        prob = ((max(rates.ni) - rates.ni)**2 + 1)/sum((max(rates.ni) - rates.ni)**2 + 1)
    else:
        prob = np.ones(rates.nbr_users)/rates.nbr_users
    for i in range(iterations):
        bucket = np.random.choice(range(distrobution.shape[0]), m, replace=False, p=prob)
        temp = sum(distrobution[bucket][:])
        var = sum((np.mean(temp) - temp)**2)
        if var < var_min:
            var_min = var
            bucket_min = bucket
    
    return bucket_min
    
        #print(bucket)
def g1(rates, dict, dataset, k, mode):
    """
    This function calculates the distribution of classes
    in the local data for every user. From this result the 
    function returns a set of m users which combined data 
    are somwhat evenly distributed over classes.

    m: Number of users to select
    dict: local data for all users, as indexes to the dataset
    dataset: Dataset of intresst 
    selectedUsers: How users has previously been selected. 
    """
    labels = dataset.train_labels.numpy()
    nbrOfClasses = 10
    dist = np.zeros((len(dict), nbrOfClasses))
    for user in range(len(dict)):
        for feature in dict[user]:
            dist[user][labels[feature]] += 1
    dist = dist/len(dict[0])
    
    return search(rates, dist, k, 1000, mode)
        
## Data Logger Class to log Data Size and Capacity for each round of training
def __init__(self, filename="data_log.csv", model_name="", dataset_name="", scheduler_model="", num_users=0, proportion_users=0.0, epoch=0):
        # ... [rest of the initialization code] ...

        # Store the additional parameters as instance attributes
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.scheduler_model = scheduler_model
        self.num_users = num_users
        self.proportion_users = proportion_users
        self.epoch = epoch


class DataLogger:
    def __init__(self, filename="data_log.csv", model_name="", dataset_name="", mode=""):
        
        # Store the additional parameters as instance attributes
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.mode = mode

        # Create the 'log' directory if it doesn't exist
        if not os.path.exists('log'):
            os.makedirs('log')
        
        # Generate a unique filename based on the current date and time
        self.current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.filename = f'log/data_log_{self.current_time}.csv'
        
        # # Initialize the CSV file with headers
        # with open(self.filename, 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(["Round", "User", "Number of Data Points", "Memory Size (MB)", "Capacity"])

        # Initialize the DataFrame with headers
        self.df = pd.DataFrame(columns=["Round", "User", "Number of Data Points", "Memory Size (MB)", "Capacity"])


    def log_data(self, round_num, user, dataset, dict_users, capacity):
        # capacity: 
        # It represents the capacity of the user's channel
        # The capacity value us the maximum number of bits that can be transmitted per symbol 
        # over the channel for each user
        # For calculation, look at the function update() in the class Rates
        # Formula: capacity = log2(1+pow(abs(h),2) * pow(self.SNR/10, 10))
        # h is the channel gain and SNR is the signal to noise ratio

        # Number of data points
        num_data_points = len(dict_users[user]) 

        # Memory size in MB of the data
        data_samples = [dataset[i] for i in dict_users[user]]
        data_tensor = torch.stack([sample[0] for sample in data_samples])
        memory_size = data_tensor.element_size() * data_tensor.nelement() / 1024 / 1024 # in MB

        # Append the data to the DataFrame
        new_data = {
            "Round": round_num, 
            "User": user, 
            "Number of Data Points": num_data_points, 
            "Memory Size (MB)": f"{memory_size:.2f}", 
            "Capacity": capacity
        }
        new_df = pd.DataFrame(new_data, index=[0])
        self.df = pd.concat([self.df, new_df], ignore_index=True)

        # Save the DataFrame to the CSV file
        self.df.to_csv(self.filename, index=False)

    def generate_summary(self):
        # 1) How many rounds already completed?
        total_rounds = self.df['Round'].max() + 1

        # 2) Summary of number of users for each round
        users_per_round = self.df.groupby('Round').size()

        # 3) Average capacity of users for each round
        avg_capacity_per_round = self.df.groupby('Round')['Capacity'].mean()

        # Write the summary to a text file
        summary_filename = f'log/data_log_{self.current_time}_summary.txt'
        
        with open(summary_filename, 'w') as file:
            # The model information
            file.write(f"Model: {self.model_name}\n")
            file.write(f"Dataset: {self.dataset_name}\n")
            file.write(f"Scheduler: {self.mode}\n")
                
            # The round information
            file.write(f"Total rounds completed: {total_rounds}\n\n")
            file.write("Number of users per round:\n")
            for round_num, count in users_per_round.items():
                file.write(f"Round {round_num}: {count} users\n")
            file.write("\nAverage capacity per round:\n")
            for round_num, avg_capacity in avg_capacity_per_round.items():
                file.write(f"Round {round_num}: {avg_capacity:.2f}\n")

