import math
import numpy as np
from scipy.optimize import fsolve
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def select_user(available_users, available_time, fluction_computation_capabilities, maximum_computation_capabilities):
    '''
     
     The user selection model may change in the future. For now, it is a random selection.

    '''
    
    return np.random.choice(list(available_users))


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
    available_time = deadline_constraint

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

    
    # Merging computation latency and interruption latency
    # if is_interruption:
    #     total_latency, available_time, latency_info = get_latency_metrics(t_cp, num_interruption, interruption_duration, interruption_interval, deadline_constraint)
    # else:
    #     total_latency = t_cp
    
    
    total_latency, available_time, latency_info = get_latency_metrics(t_cp, num_interruption, interruption_duration, interruption_interval, deadline_constraint)

    # If the total latency is less than the deadline constraint, then the task can be completed within the deadline
    if total_latency <= deadline_constraint:        
        return True, available_time, latency_info
    else:
        return False, available_time, latency_info


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
        return False, int(0), 0.0, 0.0, 
    
##############################################################################################################
### communication latency



##############################################################################################################
### Total latency

def get_latency_metrics(t_cp, num_interruption, interruption_duration, interruption_interval, deadline_constraint):

    latency_info = compute_latency_info(t_cp, num_interruption, interruption_duration, interruption_interval, deadline_constraint)

    total_latency = sum(latency_info['computation_segments']) + sum(latency_info['interruption_segments'])
    available_time = get_available_time(latency_info, total_latency, deadline_constraint)

    bars, colors = generate_bars_and_colors(latency_info)
    visualize_latency(bars, colors, deadline_constraint)

    
    # for i in range(num_interruption):
    #     # Time until the next interruption
    #     next_interval = interruption_interval[i] if i < len(interruption_interval) else remaining_time
    #     added_duration = min(next_interval, remaining_time)

    #     bars.append(added_duration)
    #     colors.append('blue')

    #     total_latency += added_duration
    #     remaining_time -= added_duration
        
    #     if remaining_time <= 0:
    #         break

    #     bars.append(interruption_duration[i])
    #     colors.append('red')

    #     total_latency += interruption_duration[i]

    # if remaining_time > 0:
    #     bars.append(remaining_time)
    #     colors.append('blue')
    #     total_latency += remaining_time  # add any remaining computation time if interruptions have been taken care of
    
    

    return total_latency, available_time, latency_info

# def compute_latency_info(t_cp, num_interruption, interruption_duration, interruption_interval, deadline_constraint):
#     elapsed_time = 0
#     remaining_computation = t_cp
#     available_time = deadline_constraint

#     latency_info = {
#         "computation_segments": [],  # Computation durations until next interruption or end
#         "interruption_segments": [],  # Each interruption duration
#         "available_times_post_interruption": []  # Available time after each interruption
#     }

#     for i in range(num_interruption):
#         # Time until the next interruption
#         next_interval = interruption_interval[i] if i < len(interruption_interval) else remaining_computation
#         computation_duration = min(next_interval, remaining_computation)

#         elapsed_time += computation_duration
#         remaining_computation -= computation_duration
#         available_time -= computation_duration

#         latency_info["computation_segments"].append(computation_duration)

#         if remaining_computation <= 0:
#             break

#         # Apply interruption
#         interrupt_duration = interruption_duration[i]
#         elapsed_time += interrupt_duration
#         available_time -= interrupt_duration

#         latency_info["interruption_segments"].append(interrupt_duration)
#         latency_info["available_times_post_interruption"].append(available_time)

#     # If there's any computation left
#     if remaining_computation > 0:
#         latency_info["computation_segments"].append(remaining_computation)

#     return latency_info


def compute_latency_info(t_cp, num_interruption, interruption_duration, interruption_interval, deadline_constraint):
    elapsed_time = 0
    remaining_computation = t_cp
    available_time = deadline_constraint

    latency_info = {
        "computation_segments": [],
        "interruption_segments": [],
        "available_times_post_interruption": [],
        "completion_possible": []
    }

    for i in range(num_interruption):
        # Time until the next interruption
        next_interval = interruption_interval[i] if i < len(interruption_interval) else remaining_computation
        computation_duration = min(next_interval, remaining_computation)

        elapsed_time += computation_duration
        remaining_computation -= computation_duration
        available_time -= computation_duration

        latency_info["computation_segments"].append(computation_duration)

        if remaining_computation <= 0:
            break

        # Apply interruption
        interrupt_duration = interruption_duration[i]
        elapsed_time += interrupt_duration
        available_time -= interrupt_duration

        latency_info["interruption_segments"].append(interrupt_duration)
        latency_info["available_times_post_interruption"].append(available_time)
        
        # Check if completion is possible post interruption
        if remaining_computation <= available_time:
            latency_info["completion_possible"].append(True)
        else:
            latency_info["completion_possible"].append(False)

    # If there's any computation left after all interruptions
    if remaining_computation > 0:
        latency_info["computation_segments"].append(remaining_computation)
        # Last check for completion possibility
        if remaining_computation <= available_time:
            latency_info["completion_possible"].append(True)
        else:
            latency_info["completion_possible"].append(False)

    return latency_info

def get_available_time(latency_info, total_latency, deadline_constraint):
    available_times = latency_info.get("available_times_post_interruption", [])

    # Check if the list is empty
    if not available_times:
        if total_latency < deadline_constraint:
            return 0
        return deadline_constraint
    
    # If all values in available_times_post_interruption are positive, then the last value is the available time
    if all(value > 0 for value in available_times):
        return available_times[-1]
    
    # If there is at least one negative value, then the last positive value is the available time
    for value in reversed(available_times):
        if value > 0:
            return value
        
    # If there is no positive value, then the available time is 0
    return 0

    # # If all values in available_times_post_interruption are positive, then the last value is the available time
    # if all(value > 0 for value in latency_info["available_times_post_interruption"]):
    #     return latency_info["available_times_post_interruption"][-1]

    # # If there is at least one negative value, then the last positive value is the available time
    # for value in reversed(latency_info["available_times_post_interruption"]):
    #     if value > 0:
    #         return value
        
    # # If there is no positive value, then the available time is 0
    # if all(value <= 0 for value in latency_info["available_times_post_interruption"]):
    #     return 0
    
    # # If latency_info["available_times_post_interruption"] is empty 
    # # but total_latency is less than deadline_constraint, then the available time is 0
    # if total_latency < deadline_constraint and latency_info["available_times_post_interruption"] is None:
    #     return 0
    
    # # If latency_info["available_times_post_interruption"] is empty and total_latency is greater than deadline_constraint,
    # # then user is not capable to start the task, so time was reserved as deadline_constraint
    # if total_latency >= deadline_constraint and latency_info["available_times_post_interruption"] is None:
    #     return deadline_constraint


##############################################################################################################

def generate_bars_and_colors(latency_info):
    bars = []
    colors = []

    for comp, inter in zip(latency_info["computation_segments"], latency_info["interruption_segments"]):
        bars.extend([comp, inter])
        colors.extend(['blue', 'red'])

    # If computation segments are more than interruption segments, add the remaining computation segments
    if len(latency_info["computation_segments"]) > len(latency_info["interruption_segments"]):
        bars.extend(latency_info["computation_segments"][len(latency_info["interruption_segments"]):])
        colors.extend(['blue'] * (len(latency_info["computation_segments"]) - len(latency_info["interruption_segments"])))

    return bars, colors


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