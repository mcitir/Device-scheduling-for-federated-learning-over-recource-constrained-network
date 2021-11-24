import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

acc1 = np.load("save/acc_fed_mnist_cnn_30_C0.1_iidTrue_snr30_compSpar_schedulingrandom.npy")
acc2 = np.load("save/acc_fed_mnist_cnn_30_C0.1_iidTrue_snr30_compSpar_schedulingcapacity.npy")
acc3 = np.load("save/acc_fed_mnist_cnn_30_C0.1_iidTrue_snr30_compSpar_schedulingg1.npy")
acc4 = np.load("save/acc_fed_mnist_cnn_30_C0.1_iidTrue_snr30_compSpar_schedulingG1-M.npy")
acc5 = np.load("save/acc_fed_mnist_cnn_30_C0.1_iidTrue_snr30_compspar_schedulingBN2.npy")

plt.figure()
plt.plot(range(len(acc1)), acc1, label='RS')
plt.plot(range(len(acc2)), acc2, label='BC')
plt.plot(range(len(acc3)), acc3, label='G1')
plt.plot(range(len(acc3)), acc4, label='G1-M')
plt.plot(range(len(acc4)), acc5, label='BN2')
plt.ylabel('Accurecy')
plt.xlabel('Rounds')
plt.legend()

plt.show()