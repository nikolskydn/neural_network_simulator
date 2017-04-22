#!/usr/bin/python
from numpy import *
import sys
nS=1
#nD=1
Ne=800
Ni=200
N=Ne+Ni
t=0
te=1000
dt=1
dtDump=500
re=random.rand(Ne)
ri=random.rand(Ni)
a=r_[0.02*ones(Ne),0.02+0.08*ri]
b=r_[0.2*ones(Ne),0.25-0.05*ri]
c=r_[-65+15*re**2,-65*ones(Ni)]
d=r_[8-6*re**2,2*ones(Ni)]
w=c_[0.5*random.rand(N,Ne), -random.rand(N,Ni)]
VR=-65
V=VR*ones(N)
U=b*V
VP=30
VR=-65
V=VR*ones(N)
m=V>=VP
U=b*V
I=r_[5*random.randn(Ne),2*random.randn(Ni)]
f=open("pcnni2003_nNeurs%i.sin" % (Ne+Ni),"w")
#f=sys.stdout
def printInFile():
    f.write("#scalar simulatorNumbers\n %i\n" % nS)
    #f.write("#scalar dataNumbers\n %i\n" % nD)
    f.write("#scalar time\n %f\n" % t)
    f.write("#scalar timeEnd\n %f\n" % te)
    f.write("#scalar deltaTime\n %f\n" % dt)
    f.write("#scalar deltaTimeForDump\n %f\n" % dtDump)
    f.write("#scalar numberOfNeurs\n %i\n" % N)
    f.write("#scalar numberOfExcitatoryNeurs\n %i\n"% Ne)
    f.write("#vector VNeurs\n")
    savetxt(f, V.reshape(1,N), fmt='%8.4f')
    f.write("#vector m\n")
    savetxt(f, m.reshape(1,N), fmt='%8d')
    f.write("#scalar VNeursPeak\n %f\n" % VP)
    f.write("#scalar VNeursReset\n %f\n" % VR)
    f.write("#vector INeurs\n")
    savetxt(f, I.reshape(1,N), fmt='%8.4f')
    f.write("#matrix weightsOfConns\n")
    #f.write("#")
    #savetxt(f,arange(0,N).reshape(1,N),fmt="%8i")
    savetxt(f, w, fmt='%8.4f')
    f.write("#vector UNeurs\n")
    savetxt(f, U.reshape(1,N), fmt='%8.4f')
    f.write("#vector aNeurs\n")
    savetxt(f, a.reshape(1,N), fmt='%8.4f')
    f.write("#vector bNeurs\n")
    savetxt(f, b.reshape(1,N), fmt='%8.4f')
    f.write("#vector cNeurs\n")
    savetxt(f, c.reshape(1,N), fmt='%8.4f')
    f.write("#vector dNeurs\n")
    savetxt(f, d.reshape(1,N), fmt='%8.4f')

printInFile()
