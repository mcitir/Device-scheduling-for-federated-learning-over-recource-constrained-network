import os
import sys
script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'utils')
sys.path.append( mymodule_dir)
import channel


num_users = 100
std_noise = 1
power = 0.1
B = 0
beta = 1
cg = channel.channel_capacity(num_users, std_noise, power, B, beta)

print("Testing")
print(cg)