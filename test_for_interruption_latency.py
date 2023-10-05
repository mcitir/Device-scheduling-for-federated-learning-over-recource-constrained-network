import numpy as np
import matplotlib.pyplot as plt
from utils.custom_utils import generate_interruption_latency

lambda_i = 0.002
deadline_constraint = 600
MU_K = 5

all_durations = []
all_intervals = []

for _ in range(500):
    is_interrupted, num_interruptions, durations, intervals = generate_interruption_latency(lambda_i, deadline_constraint, MU_K)
    if is_interrupted:
        all_durations.extend(durations)
        all_intervals.extend(intervals)


plt.figure(figsize=(18, 6))

# Sorted values
sorted_durations = sorted(all_durations)
sorted_intervals = sorted(all_intervals)

# Interruption durations
plt.subplot(1, 2, 1)
plt.scatter(sorted_durations, range(len(sorted_durations)),color='blue', s=10)
plt.title('Interruption Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Index')

# Interruption intervals
plt.subplot(1, 2, 2)
plt.scatter(sorted_intervals, range(len(sorted_intervals)), color='blue', s=10)
plt.title('Interruption Intervals')
plt.xlabel('Interval (seconds)')
plt.ylabel('Index')

plt.tight_layout()
plt.show()



