import math
import numpy as np
from scipy.optimize import fsolve
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def can_complete_task(fluction_computation_capability, maximum_computation_capability, deadline_constraint, datasize, LAMBDA_I, MU_K):
    """
    Parameters:
    -   fluction_computation_capability: float
    -   maximum_computation_capability: float
    -   deadline_constraint: float
    -   datasize: float
    -   LAMBDA_I: float
    -   MU_K: float

    Returns:
    -   bool: True if the task can be completed within the deadline, False otherwise
    """

    # calculate t_cp via generate_computation_latency()
    u_cp, t_cp = generate_computation_latency(fluction_computation_capability, 
                                                           maximum_computation_capability, 
                                                           datasize)
    print("u_cp: ", u_cp)
    print("t_cp: ", t_cp)

    # generate interruption latency
    is_interruption, num_interruption, interruption_duration, interruption_interval = generate_interruption_latency(LAMBDA_I, 
                                                                                                                    deadline_constraint, 
                                                                                                                    MU_K)

    print("is_interruption: ", is_interruption)
    print("num_interruption: ", num_interruption)
    print ("interruption_duration: ", interruption_duration)
    print ("interruption_interval: ", interruption_interval)
    
    # Merging computation latency and interruption latency
    if is_interruption:
        total_latency = calculate_total_latency(t_cp, num_interruption, interruption_duration, interruption_interval, deadline_constraint)
    else:
        total_latency = t_cp    

    # If the total latency is less than the deadline constraint, then the task can be completed within the deadline
    if total_latency <= deadline_constraint:
        return True


##############################################################################################################
### computation latency

def exponential_inverse(u, mu_i, D_i, a_i):

    def fun(t):
        return 1 - np.exp(-((mu_i * t) / D_i) + mu_i * a_i) - u
    
    # Initial guess
    t_sample = fsolve(fun, 0.0)

    return t_sample[0]

def generate_computation_latency(fluction_computation_capability, maximum_computation_capability, datasize):
    
    """
    
    This function generates a random sample from the distribution of computation latency
    
    """

    # u should be a random number between 0 and 1
    u = np.random.uniform(0, 1)

    t_sample = exponential_inverse(u, fluction_computation_capability, datasize, maximum_computation_capability)
    return u, t_sample

##############################################################################################################
### interuption latency

def generate_interruption_latency(lambda_i,deadline_constraint, MU_K):
     
    num_interruption = np.random.poisson(lambda_i*deadline_constraint) # lambda_i * deadline_constraint: number of interruptions in a round time
    if num_interruption> 0: # if there is an interruption
        print("Number of interruption: ", num_interruption)

        interruption_duration = np.random.exponential(scale=MU_K, size=num_interruption) # duration for each interruption
        print("interruption duration: ", interruption_duration)

        interruption_interval = np.random.exponential(scale=1/lambda_i, size=num_interruption) # interval between two interruptions
        print("interruption interval: ", interruption_interval)
        return True, num_interruption, interruption_duration, interruption_interval
    else:
        print("No interruption")
        return False, 0.0, 0.0, 0.0
    
##############################################################################################################
### communication latency



##############################################################################################################
### Total latency

# def calculate_total_latency(t_cp, num_interruption, interruption_duration, interruption_interval):
#     total_latency = 0
    
#     # Time before the first interruption
#     if num_interruption > 0:
#         total_latency += interruption_interval[0]
#         remaining_time = t_cp - interruption_interval[0]  # kesintiden sonra kalan süre
    
#     else:
#         remaining_time = t_cp

#     for i in range(num_interruption):
#         # Adding the duration of the interruption
#         total_latency += interruption_duration[i]
        
#         # If there is another interruption, add the interval between the two interruptions
#         if i < num_interruption - 1:
#             interval = min(remaining_time, interruption_interval[i+1])  # interval between two interruptions or remaining time (whichever is shorter)
#             total_latency += interval
#             remaining_time -= interval
        
#         # If there is no interruption after this one, add the remaining time
#         else:
#             total_latency += remaining_time
            
#     return total_latency

def calculate_total_latency(t_cp, num_interruption, interruption_duration, interruption_interval, deadline_constraint):
    total_latency = 0
    remaining_time = t_cp

    bars = []
    colors = []
    
    for i in range(num_interruption):
        # Time until the next interruption
        next_interval = interruption_interval[i] if i < len(interruption_interval) else remaining_time
        #added_duration = min(next_interval + interruption_duration[i], remaining_time)
        added_duration = min(next_interval, remaining_time)

        bars.append(added_duration)
        colors.append('blue')

        total_latency += added_duration
        remaining_time -= added_duration
        
        if remaining_time <= 0:
            break

        bars.append(interruption_duration[i])
        colors.append('red')

        total_latency += interruption_duration[i]
        #remaining_time -= interruption_duration[i]

        # if remaining_time <= 0:
        #     break

        # if remaining_time <= 0:
        #     return total_latency
    if remaining_time > 0:
        bars.append(remaining_time)
        colors.append('blue')
        total_latency += remaining_time  # add any remaining computation time if interruptions have been taken care of
    
    visualize_latency(bars, colors, deadline_constraint)

    return total_latency


##############################################################################################################

def visualize_latency(bars, colors, deadline_constraint):
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 2))

    left = 0

    for bar_width, color in zip(bars, colors):
        ax.barh('Total Latency', bar_width, color=color, left=left, height=0.1, align='center')

        left += bar_width
    
    ax.axvline(x=deadline_constraint, color='orange', linestyle='--', linewidth=1.5)
    #ax.text(deadline_constraint, 0.12, str(deadline_constraint), color='green', ha='center')

    ax.set_xlabel('Time')

    plt.show(block=False)
    plt.pause(3)

    
    




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


# def can_complete_task(fluction_computation_capability, maximum_computation_capability, deadline_constraint, datasize, LAMBDA_I, MU_K):
#     """
#     Parameters:
#     -   fluction_computation_capability: float
#     -   maximum_computation_capability: float
#     -   deadline_constraint: float
#     -   datasize: float
#     -   theshold: float

#     Returns:
#     -   bool: True if the task can be completed within the deadline, False otherwise
#     """
    ##################################################
    # Because of implementation of t_sample generation, the old version of this function is not used anymore

    # computation_latency_probability = compute_computation_latency_probability(fluction_computation_capability,
    #                                                       maximum_computation_capability,
    #                                                       deadline_constraint, datasize)
    # interruption_latency_probability = compute_interruption_latency_probability()
    # communication_latency_probability = compute_communication_latency_probability()

    # if computation_latency_probability != -1:

    #     total_latency_probability = computation_latency_probability * interruption_latency_probability * communication_latency_probability
        
    #     return np.random.choice([True, False], p=[1-total_latency_probability, total_latency_probability])
    # else:
    #     # A log record will be created when this exception is raised
    #     raise ValueError("Error in probability calculation")
    ##################################################