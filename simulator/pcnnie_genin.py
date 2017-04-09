#!/usr/bin/python
from numpy import *
import sys
nS=1
Ne=800
Ni=200
t=0
tEnd=1000
dt=1
re=random.rand(Ne)
ri=random.rand(Ni)
a=r_[0.02*ones(Ne),0.02+0.08*ri]
b=r_[0.2*ones(Ne),0.25-0.05*ri]
c=r_[-65+15*re**2,-65*ones(Ni)]
d=r_[8-6*re**2,2*ones(Ni)]
w=c_[0.5*random.rand(Ne+Ni,Ne), -random.rand(Ne+Ni,Ni)]
VR=-65
V=VR*ones(Ne+Ni)
U=b*V
VP=30
VR=-65
V=VR*ones(Ne+Ni)
V[0]=V[2]=-50
VP=30
m=V>=VP
U=b*V
I=r_[5*random.randn(Ne),2*random.randn(Ni)]

def printInFile():
    print "# nS\n%i" % nS
    print "# t\n%f" % t
    print "# tEnd\n%f" % tEnd
    print "# dt\n%f" % dt
    print "# dtDump\n%f" % dt
    print "# nNeurs\n%i" % (Ni+Ne)
    print "# nNeursExc\n%i"% Ne
    print "# V"
    savetxt(sys.stdout, V, fmt='%8.4f', delimiter=' ', newline=' ')
    print "\n# m"
    savetxt(sys.stdout, m, fmt='%8d',delimiter=' ', newline=' ')
    print "\n# VPeak\n %f" % VP
    print "\n# VReset\n %f" % VR
    print "\n# I"
    savetxt(sys.stdout, I, fmt='%8.4f', delimiter=' ', newline=' ')
    print "\n# w"
    savetxt(sys.stdout, w, fmt='%8.4f', delimiter=' ', newline=' ')
    print "\n# U"
    savetxt(sys.stdout, U, fmt='%8.4f', delimiter=' ', newline=' ')
    print "\n# a"
    savetxt(sys.stdout, a, fmt='%8.4f', delimiter=' ', newline=' ')
    print "\n# b"
    savetxt(sys.stdout, b, fmt='%8.4f', delimiter=' ', newline=' ')
    print "\n# c"
    savetxt(sys.stdout, c, fmt='%8.4f', delimiter=' ', newline=' ')
    print "\n# d"
    savetxt(sys.stdout, d, fmt='%8.4f', delimiter=' ', newline=' ')

printInFile()
