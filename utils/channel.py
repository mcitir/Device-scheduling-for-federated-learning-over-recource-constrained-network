from numpy import log2, sqrt, zeros, transpose
from numpy.random import randn, random, standard_normal, uniform

def channel_capacity(num_users, noise_power, power, B):
    """
    num_users: number of users

    std_noise: standard deviation of addetive noise

    power: Power of transmitted signal

    B: Bandwidth

    beta: large scale fading coefficent

    """
    # Channel gain, based on relight fading, rich multipath
    beta = 1
    # h = CN(0, 1), a.k.a relight fading
    h = (standard_normal(num_users) + 1j * standard_normal(num_users)) * 1 * 0.5
    g = sqrt(beta)*h
    # capacity = log2(1+|g|^2 SNR)
    # SNR = q/N0
    # std_noise^2 = No/2 
    cg = log2(1 + (abs(g)**2 * power/(noise_power)))

    return cg

def hexagon(nbr_users, size=500):
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
    user_pos = hexagon(nbr_users)
    lengths = sqrt(user_pos[0,:]**2 +  user_pos[1,:]**2 )
    return sqrt(height**2 + lengths**2)

def largeScaleCoefficents(nbr_users):
    kb = 1.38*10**(-23)
    T = 300 # Kelvin
    n0 = kb*T
    # @TODO generate betas as a function of distances 
    # With resonable constants
    