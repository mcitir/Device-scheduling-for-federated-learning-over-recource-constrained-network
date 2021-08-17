from numpy import log2
from numpy.random import randn, random, standard_normal

def channel_capacity(num_users, std_noise, power, B, beta):
    """
    num_users: number of users

    std_noise: standard deviation of addetive noise

    power: Power of transmitted signal

    B: Bandwidth

    beta: variance of channel gain

    """
    # Channel gain, based on relight fading, rich multipath

    # g = CN(0, beta)
    g = ((standard_normal(num_users) + 1j * standard_normal(num_users)) * 
        beta * 0.5)

    # capacity = log2(1+|g|^2 SNR)
    # SNR = q/N0
    # std_noise^2 = No/2 
    cg = log2( (1 + abs(g)**2 * power/(std_noise^2)))

    return cg

