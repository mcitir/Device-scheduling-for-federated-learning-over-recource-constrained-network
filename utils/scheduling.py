import numpy as np
from numpy.random import standard_normal
from numpy import abs, log2, zeros

### Scheduler Class
class Scheduler:
    def __init__(self, m, dict_users, dataset_train, SNR, mode='old', comp='None'):
        self.rates = Rates(m, SNR)
        self.mode = mode
        self.picks = zeros(m)
        self.sinceLastPick = zeros(m)
        self.comp = comp
        self.dict_users = dict_users
        self.dataset_train = dataset_train

    def newUsers(self, k):
        self.rates.update()
        users = pick(self.rates, k, self.sinceLastPick, self.mode, self.dict_users, self.dataset_train)
        self.rates.ni[users] += 1
        self.picks[users] += 1
        self.sinceLastPick += 1
        self.sinceLastPick[users] = 0
        ratio = compressRatio(self.rates, users)
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
        


