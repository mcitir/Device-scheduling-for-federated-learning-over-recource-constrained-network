import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

# acc1 = np.load("save/acc_fed_mnist_cnn_30_C0.1_iidTrue_snr30_compQuant_schedulingrandom.npy")
acc1 = np.load("save/acc_fed_mnist_cnn_30_C0.1_iidTrue_snr30_compSpar_schedulingcapacity.npy")
acc2 = np.load("save/acc_fed_mnist_cnn_30_C0.1_iidTrue_snr30_compSpar_schedulingrandom.npy")

plt.figure()
plt.plot(range(len(acc1)), acc1, label='capacity')
plt.plot(range(len(acc2)), acc2, label='random')
plt.ylabel('Accurecy')
plt.xlabel('Rounds')
plt.legend()

plt.show()