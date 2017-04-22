#!/usr/bin/python
# -*- coding: UTF-8 -*
from sys import argv, exit 
from numpy import *
import matplotlib.pyplot as plt
if len(argv) > 1: 
    fileName=argv[1]
else:
    print "Select file with spikes:"
    print "./viewSpikes <fileName>"
    exit(1)
spikes = loadtxt(fileName)
plt.title('Spike raster diagram')
plt.xlabel('time, ms')
plt.ylabel('number')
plt.plot(spikes[:,0],spikes[:,1],'.')
plt.show()
