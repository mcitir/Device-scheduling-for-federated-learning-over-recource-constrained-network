def can_complete_task(user_capacity, task_size):
    """
    Parameters:
    - user_capacity: 
    - task_size: S

    Returns:
    - True if user can complete the task, False otherwise
    """
    
        
    return user_capacity >= task_size

def random_delay():
    
    return 1