#!/usr/bin/python
# -*- coding: UTF-8 -*
from sys import argv, exit 
from numpy import *
import matplotlib.pyplot as plt
if len(argv) > 2: 
    fileName = argv[1]
    idx = argv[2]
else:
    print "Select file with oscillograms and index:"
    print "./viewOscillograms <fileName> idxNeuron"
    exit(1)
oscgs = loadtxt(fileName)
mask=oscgs[:,:]>30
mask[:,0]=False
oscgs[mask]=20
plt.grid(True)
plt.title('Oscillogram')
plt.xlabel('time $t$, ms')
plt.ylabel('membrane potential $V$, mV')
if idx=='all':
    for i in range(1,oscgs.shape[1]):
        plt.plot(oscgs[:,0],oscgs[:,i],'-')
else:
    plt.plot(oscgs[:,0],oscgs[:,idx],linewidth=1,color='green',linestyle='dashed',label='$V_{%s}$'%idx)
Va=average(oscgs[:,1:],axis=1)
plt.plot(oscgs[:,0],Va,linewidth=2,color='blue',linestyle='solid',label='$<V>$')
plt.legend(loc='best')
plt.show()
