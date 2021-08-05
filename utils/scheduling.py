import numpy as np

def search(distrobution, m, iterations, selectedUsers):
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
    prob = ((max(selectedUsers) - selectedUsers)**2 + 1)/sum((max(selectedUsers) - selectedUsers)**2 + 1)
    for i in range(iterations):
        bucket = np.random.choice(range(distrobution.shape[0]), m, replace=False, p=prob)
        temp = sum(distrobution[bucket][:])
        var = sum((np.mean(temp) - temp)**2)
        if var < var_min:
            var_min = var
            bucket_min = bucket
    
    return bucket_min
    
        #print(bucket)
def userSelection(m, dict, dataset, selectedUsers, scheduling=True):
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
    
    return search(dist, m, 1000, selectedUsers)
        