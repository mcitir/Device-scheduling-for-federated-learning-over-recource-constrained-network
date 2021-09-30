from numpy.random import standard_normal
from numpy import abs, log2

class Rates:
    def __init__(self, nbr_users, SNR=20):
        ## OFDM Constants
        Ts = (1.0/14) * 10**(-3)
        Tu = (1.0/15) * 10**(-3)
        Tcp = (1.0/(14*15)) * 10**(-3)
        Bs = 15*10**3
        Bc = 210*10**3
        N_smooth = 14
        T_slot = 2*10**(-3)
        N_slot = 28

        ## Usefull stuff
        self.network_size=10**3
        self.nbr_users = nbr_users
        self.samples = N_smooth*N_slot
        self.sampleSize = Ts * Bs # Maybe Tu?

        ## Channel
        self.SNR = SNR
        self.update()
    
    def update(self):
        """
        Only change small scale fading, then update values. 
        """
        h = (standard_normal(self.nbr_users) + 1j * standard_normal(self.nbr_users)) * 1 * 0.5
        capacity = log2(1+pow(abs(h),2) * pow(self.SNR/10, 10))
        self.bitsPerSymbol = capacity*self.sampleSize  



def evenDistrobution(self, users_choosen = 10):
    """
    Every user gets the same amount of symbols
    """
    ### We need scheduling here? 
    samplesPerUser = round(self.samples / self.nbr_users)
    dataPerUser = self.bitsPerSymbol* samplesPerUser
    return dataPerUser


    