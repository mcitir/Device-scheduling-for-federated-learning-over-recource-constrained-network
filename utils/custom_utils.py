import math
import numpy as np
from scipy.optimize import fsolve

def can_complete_task(fluction_computation_capability, maximum_computation_capability, deadline_constraint, datasize):
    """
    Parameters:
    -   fluction_computation_capability: float
    -   maximum_computation_capability: float
    -   deadline_constraint: float
    -   datasize: float
    -   theshold: float

    Returns:
    -   bool: True if the task can be completed within the deadline, False otherwise
    """
    
    computation_latency_probability = compute_computation_latency_probability(fluction_computation_capability,
                                                          maximum_computation_capability,
                                                          deadline_constraint, datasize)
    interruption_latency_probability = compute_interruption_latency_probability()
    communication_latency_probability = compute_communication_latency_probability()

    if computation_latency_probability != -1:

        total_latency_probability = computation_latency_probability * interruption_latency_probability * communication_latency_probability
        
        return np.random.choice([True, False], p=[1-total_latency_probability, total_latency_probability])
    else:
        # A log record will be created when this exception is raised
        raise ValueError("Error in probability calculation")


def compute_computation_latency_probability(fluction_computation_capability, maximum_computation_capability, deadline_constraint, datasize):
    """
    Parameters:
    -   fluction_computation_capability: float
    -   maximum_computation_capability: float
    -   deadline_constraint: float
    -   datasize: float

    Returns:
    -   probability_of_computation_latency: float
    """
    computation_latency_probability = 1 - math.exp((-(fluction_computation_capability * deadline_constraint) / datasize) 
                            + (fluction_computation_capability * maximum_computation_capability))
    
    if 0 <= computation_latency_probability <= 1:
        return computation_latency_probability
    else:
        raise ValueError("Error in computation_latency_probability calculation")
    

def compute_interruption_latency_probability():
    """
    Parameters:

    
    Returns:

    """
    
    return 1

def compute_communication_latency_probability():
    """
    Parameters:

    
    Returns:

    """
    
    return 1


def  apply_preemption(users, probability, max_preemption_time=60):

    preemption_times = np.random.randint(1, max_preemption_time+1, size=len(users))
    apply_preemption = np.random.rand(len(users)) < probability

    # If apply_preemption is True, then preemption_times is multiplied by preemption_times, otherwise it is multiplied by 0 (no preemption)
    preemption_times *= apply_preemption
    
    return list(zip(users, preemption_times))

##############################################################################################################

def exponential_inverse(u, mu_i, D_i, a_i):

    # Function to find the root of ..
    def fun(t):
        return 1 - np.exp(-((mu_i * t) / D_i) + mu_i * a_i) - u
    
    # Initial guess
    t_sample = fsolve(fun, 0.0)

    return t_sample[0]

def generate_t_sample_for_computation_latency(fluction_computation_capability, maximum_computation_capability, datasize):
    
    """
    
    This function generates a random sample from the distribution of computation latency
    
    """

    # u should be a random number between 0 and 1
    u = np.random.uniform(0, 1)

    t_sample = exponential_inverse(u, fluction_computation_capability, datasize, maximum_computation_capability)
    # return u and t_sample
    return u, t_sample