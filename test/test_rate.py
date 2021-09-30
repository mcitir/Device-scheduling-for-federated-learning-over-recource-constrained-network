import os
import sys
script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'utils')
sys.path.append( mymodule_dir)
net_dir = os.path.join( script_dir, '..', 'models')
sys.path.append( net_dir)
from rate import Rates
from scheduling import Scheduler
from Nets import CNNMnist
from options import args_parser

import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from numpy import average

scheduler = Scheduler(100, 10)
picked = scheduler.pick()
print(picked)





#args = args_parser()
#cnn = CNNMnist(args=args)
#optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

#sum = 0
#for para in cnn.parameters():
    # sum += torch.numel(para)
    # print(type(para[0][0][0][0].item()))

#C = Rates(10)
#print(average(C.bitsPerSymbol))

#ratio = C.compressionRatio()
#print(str(ratio))

## 22340
## Torch usually uses float32 = 32 bits. => Huge number