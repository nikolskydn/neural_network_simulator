#!/usr/bin/python
# -*- coding: UTF-8 -*
from sys import argv, exit 
from numpy import *
import matplotlib.pyplot as plt
if len(argv) > 2: 
    fileName = argv[1]
    idx = argv[2]
else:
    print "Select file with data and elem. index:"
    print "./viewCa.py <fileName> idxAstr"
    exit(1)
oscgs = loadtxt(fileName)
mask=oscgs[:,:]>30
mask[:,0]=False
oscgs[mask]=20
plt.grid(True)
plt.title('Ca oscillogram')
plt.xlabel('time $t$, ms')
plt.ylabel('concentration $\\nu_{Ca}, \\nu m$')
plt.plot(oscgs[:,0],oscgs[:,idx],linewidth=1,color='red',linestyle='dashed',label='$\\nu_{Ca %s}$'%idx)
Va=average(oscgs[:,1:],axis=1)
plt.plot(oscgs[:,0],Va,linewidth=2,color='green',linestyle='solid',label='$<\\nu_{Ca}>$')
plt.legend(loc='best')
plt.show()
