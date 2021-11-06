import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

loss_train = np.load("fed_mnist_cnn_5_C0.1_iidTrue.npy")

plt.figure()
plt.plot(range(len(loss_train)), loss_train)
plt.ylabel('train_loss')

plt.show()