from utils.custom_utils import generate_computation_latency
import matplotlib.pyplot as plt

UNIFORM_DATASET_SIZE = 5000
MAX_COMPUTATION = 0.08
FLUCTUATION_COMPUTATION = 30

# generet 500 samples from the distribution of computation latency and store u and t_sample in a list
u_list = []
t_sample_list = []

for i in range(500):
    u, t_sample = generate_computation_latency(FLUCTUATION_COMPUTATION, MAX_COMPUTATION, UNIFORM_DATASET_SIZE)
    u_list.append(u)
    t_sample_list.append(t_sample)

# plot as line graph the distribution of computation latency against t_sample (X axis) u (Probability) (Y axis)

plt.scatter(t_sample_list, u_list, color='blue', s=10)
plt.xlabel('Computation Latency')
plt.ylabel('Probability')
plt.title('Distribution of Computation Latency')
plt.show()


