import os
import sys
script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'utils')
sys.path.append( mymodule_dir)
import channel
import matplotlib.pyplot as plt

num_users = 100
B = 10**5
cg = channel.channel_capacity(num_users, B)


print("Testing")

#x = channel.hexagon(1000*num_users)
#print(x)
#plt.scatter(x[0], x[1])
#plt.show()

#y = channel.distances(1000*num_users)
#print(y)
#plt.hist(y, bins=200)
#plt.show()

print(cg)