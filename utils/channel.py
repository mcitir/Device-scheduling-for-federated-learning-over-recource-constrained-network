from numpy import log2, log10, sqrt, zeros, transpose
from numpy.random import randn, random, standard_normal, uniform


##################################################################
### Coherence, for now only pedestrians indoors! #################
### BASED ON "FUNDEMENTALS OF MASSIVE MIMO" chapter 2 ############

def coherenceInterval():            # Also known as coherence block
    tc = coherenceTime()
    bc = coherenceBandwidth()
    return tc*bc

def coherenceTime():
    fc = 2 * 10**9                  # Close to the avarage for 4G
    v = 1.5                         # avarage velocity [m/s] of users
    tc = (3 * 10**8 / fc) / (2 * v) # Coherence time
    return tc

def coherenceBandwidth():
    d1_d2 = 30                      # Distance from user to wall
    bc = 3 * 10**8 / d1_d2
    return bc


##################################################################
### CHANNEL MODELL ###############################################
def channel_capacity(nbr_users, B):
    """
    nbr_users: number of users

    B: Bandwidth

    returns cg: randomly generates [bits/s/Hz] for 'nbr_users' based on 
    a physical model within a cell. 
    """
    ###########################################################################
    ##### Noise power and antenna gain, based on lab2 TSKS14 ##################
    kb = 1.38*10**(-23)
    T = 300 # Kelvin
    n0 = kb*T
    n0_dBm = 10*log10(n0*B/(10**(-3)));                             #-143
    print("No: " + str(n0_dBm))

    uplinkRadPower = 20                                             # dBm
    antennaGain = 3                                                 # dBi
    baseGain = 3                                                    # dbi
    noiseFigure = 10                                                # db
    effektNoise = (n0_dBm + noiseFigure)                            # dBm
    pul_db = uplinkRadPower + antennaGain + baseGain - effektNoise  # dB
    pul = 10**(pul_db/10)   # ca 140 dB total uplink gain

    ############################################################################
    ### Channel gain, beta based on TSKS lab 2, and h based on relight fading ##
    h = (standard_normal(nbr_users) + 1j * standard_normal(nbr_users)) * 1 * 0.5
    beta = largeScaleCoefficents(nbr_users)
    g = sqrt(beta)*h
    
    ############################################################################
    ### Capacity [bits/s/Hz] ###################################################
    cg = log2(1 + (abs(g)**2 * pul))
    return cg

def hexagon(nbr_users, size=500):
    """
    Randomly generate 'nbr_users' points wihin a hexagon, where the maximum 
    distance from the center of the hexagon is 'size'. 
    """
    x = zeros((2, nbr_users))
    for i in range(nbr_users):
        r = uniform(0,1)
        uv = [uniform(0,1), uniform(0,1)]

        if r <= 1/3:
            x[0,i] = sqrt(3)*uv[0]/2
            x[1,i] = -uv[0]/2 + uv[1]
        elif (r > 1/3 and r <= 2/3):
            x[0,i] = -sqrt(3)*uv[0]/2 + sqrt(3)*uv[1]/2
            x[1,i] = -uv[0]/2 - uv[1]/2
        else:
            x[0,i] = -sqrt(3)*uv[1]/2
            x[1,i] = uv[0] - uv[1]/2

    x *= size
    return x

def distances(nbr_users, size=500, height=35):
    """
    Help function to calculate the distance from the user to the BS.
    It also assumes that the antenna is located 35m above the users. 
    This is more realistic, but also makes the beta modell more robust
    since it breaks down for small distances. 
    """
    user_pos = hexagon(nbr_users)
    lengths = sqrt(user_pos[0,:]**2 +  user_pos[1,:]**2 )
    return sqrt(height**2 + lengths**2)

def largeScaleCoefficents(nbr_users):
    """
    This function randomly generates 'nbr_users' large scale fading coefficents

    Based on TSKS14 lab 2. 
    The PM said that β[dB] =−17−37.6 log10(d) is a simple way
    to modell the channle variance.
    """
    beta_dB = -17 -37.6*log10(distances(nbr_users)) # [dB]
    return 10**(beta_dB/10)
    