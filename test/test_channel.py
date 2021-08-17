import os
import sys
script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'utils')
sys.path.append( mymodule_dir)
import channel
import matplotlib.pyplot as plt

num_users = 100
noise_power = 0.01
power = 1
B = 10**5
cg = channel.channel_capacity(num_users, noise_power, power, B)


print("Testing")

x = channel.hexagon(num_users)
#print(x)
#plt.scatter(x[0], x[1])
#plt.show()

y = channel.distances(1000*num_users)
print(y)
plt.hist(y, bins=200)
plt.show()

# print(cg)