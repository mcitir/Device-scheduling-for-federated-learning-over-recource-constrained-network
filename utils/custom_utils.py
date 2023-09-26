import math

def can_complete_task(fluction_computation_capability, maximum_computation_capability, deadline_constraint, datasize, threshold):
    """
    Parameters:
    -   fluction_computation_capability: float
    -   maximum_computation_capability: float
    -   deadline_constraint: float
    -   datasize: float
    -   theshold: float

    Returns:
    -   bool: True if the probability of computation latency is greater than the threshold
    """

    computation_latency_probability = compute_computation_latency_probability(fluction_computation_capability,
                                                          maximum_computation_capability,
                                                          deadline_constraint, datasize)
    if computation_latency_probability != -1:
        return 1 - computation_latency_probability >= threshold
    else:
        # Burada bir log kaydı alabilir veya exception fırlatabilirsiniz
        raise ValueError("Error in compute_computation_latency_probability")


def compute_computation_latency_probability(fluction_computation_capability, maximum_computation_capability, deadline_constraint, datasize):
    """
    Parameters:
    -   fluction_computation_capability: float
    -   maximum_computation_capability: float
    -   deadline_constraint: float
    -   datasize: float

    Returns:
    -   float: Probability of computation latency
    """
    exponential = math.exp((-(fluction_computation_capability * deadline_constraint) / datasize) 
                            + (fluction_computation_capability * maximum_computation_capability))
    
    if 0 <= exponential <= 1:
        return (1 - exponential)
    else:
        return -1
    

def random_delay():
    
    return 1