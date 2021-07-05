import numpy as np

def search(distrobution, m, iterations):
    bucket = np.zeros(distrobution.shape[0])
    bucket[:m] = 1
    mse_min = float('inf')
    bucket_min = np.zeros(distrobution.shape[0])
    for i in range(iterations):
        np.random.shuffle(bucket)
        temp = sum(distrobution[np.where(bucket == 1)][:])
        mse = sum((np.mean(temp) - temp)**2)
        if mse < mse_min:
            mse_min = mse
            bucket_min = bucket
    
    #print(mse_min)
    return np.where(bucket_min == 1)[0]
    
        #print(bucket)
def userSelection(m, dict, dataset):
    labels = dataset.train_labels.numpy()
    nbrOfClasses = 10
    # Distobution for users
    dist = np.zeros((len(dict), nbrOfClasses))
    for user in range(len(dict)):
        for feature in dict[user]:
            dist[user][labels[feature]] += 1
    dist = dist/len(dict[0])
    
    return search(dist, m, 1000)
    print(reslut)
        