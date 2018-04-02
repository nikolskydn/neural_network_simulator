#!/usr/bin/python
# -*- coding: UTF-8 -*
from sys import argv, exit 
from numpy import *
import matplotlib.pyplot as plt
if len(argv) > 1: 
    fileName = argv[1]
else:
    print "Select file with data:"
    print "./viewOscillograms.py <fileName> "
    exit(1)
oscgs = loadtxt(fileName)
mask=oscgs[:,:]>30
mask[:,0]=False
oscgs[mask]=20
plt.grid(True)
plt.title('Membrane potential oscillogram')
plt.xlabel('time $t$, ms')
plt.ylabel('Potential $V$, mV')
idx=5
plt.plot(oscgs[:,0],oscgs[:,idx],linewidth=1,color='red',linestyle='solid',label='$V_{%s}$'%idx)
idx=10
plt.plot(oscgs[:,0],oscgs[:,idx],linewidth=1,color='blue',linestyle='solid',label='$V_{%s}$'%idx)
idx=11
plt.plot(oscgs[:,0],oscgs[:,idx],linewidth=1,color='green',linestyle='solid',label='$V_{%s}$'%idx)
idx=12
plt.plot(oscgs[:,0],oscgs[:,idx],linewidth=1,color='cyan',linestyle='solid',label='$V_{%s}$'%idx)
plt.legend(loc='best')
plt.show()
