#!/usr/bin/python

print "Generating .sin file ../tests/realModel_UNN270117.sin"
print "You can use this file as in-file for cpu simulator"

f = open('../tests/realModel_UNN270117.sin', 'w')

f.write("""
#scalar nS
2
#scalar t
0.000000
#scalar tEnd
10000
#scalar dt
0.1
#scalar dtDump
1000000
#scalar nNeurs
8
#scalar nNeursExc
7
#scalar VPeak
30.000000
#scalar VReset
-65.000000

#scalar Nastr
8

#scalar Cm
1
#scalar g_Na
120
#scalar g_K 
36
#scalar g_leak
0.3
#scalar Iapp
5.2
#scalar E_Na
55
#scalar E_K 
-77
#scalar E_L 
-54.4
#scalar Esyn
-90
#scalar thetaSyn
0
#scalar kSyn
0.2
#scalar alphaGlu
0.01
#scalar alphaG
0.025
#scalar bettaG
0.5

#scalar tauIP3
7142.85714
#scalar IP3ast
0.16
#scalar a2
0.00014
#scalar d1
0.13
#scalar d2
1.049
#scalar d3
0.9434
#scalar d5
0.082
#scalar dCa
0.000001
#scalar dIP3
0.00012
#scalar c0
2
#scalar c1
0.185
#scalar v1
0.006
#scalar v4
0.0003
#scalar alpha
0.8
#scalar k4
0.0011
#scalar v2
0.00011
#scalar v3
0.0022
#scalar k3
0.1
#scalar v5
0.000025
#scalar v6
0.0002
#scalar k2
1
#scalar k1
0.0005

#scalar IstimAmplitude
1.5
#scalar IstimFrequency
1
#scalar IstimDuration
1

#matrixname wAstrNeurs
#matrix a1 a2 a3 a4 a5 a6 a7 a8
3   0   0   0   0   0   0   0
0   3   0   0   0   0   0   0
0   0   3   0   0   0   0   0
0   0   0   3   0   0   0   0
0   0   0   0   3   0   0   0
0   0   0   0   0   3   0   0
0   0   0   0   0   0   3   0
0   0   0   0   0   0   0   3

#matrixname astrConns
#matrix 1 2 3 4 5 6 7 8
0   1   0   0   1   0   0   0
1   0   1   0   0   1   0   0
0   1   0   1   0   0   1   0
0   0   1   0   0   0   0   1
1   0   0   0   0   1   0   0
0   1   0   0   1   0   0   0
0   0   1   0   0   1   0   1
0   0   0   1   0   0   1   0

#matrixname wConns
#matrix 1 2 3 4 5 6 7 8
-0.05   0.0 0.0 0.0 0.0 -0.05   0.0 0.0
-0.05   0.0 0.0 0.0 0.05    -0.05   0.0 0.0
-0.05   0.05    0.0 0.0 0.0 0.0 0.0 0.0
-0.05   0.0 0.0 0.0 0.0 0.0 -0.05   0.0
-0.05   0.0 0.0 0.0 0.0 0.0 0.0 -0.05
-0.05   0.0 -0.05   -0.05   0.0 0.0 0.0 0.0
-0.05   0.05    0.0 0.0 0.0 0.0 0.0 0.0
-0.05   0.05    -0.05   0.0 0.0 0.0 0.0 0.0

#vector spikeMask
0   0   0   0   0   0   0   0

#vector VNeurs
-75.45828   -75.45828   -75.45828   -75.45828   -75.45828   -75.45828   -75.45828   -75.45828

#vector INeurs
0   0   0   0   0   0   0   0

#vector m
0.0224  0.0224  0.0224  0.0224  0.0224  0.0224  0.0224  0.0224

#vector h
0.13517 0.13517 0.13517 0.13517 0.13517 0.13517 0.13517 0.13517

#vector n
0.68632 0.68632 0.68632 0.68632 0.68632 0.68632 0.68632 0.68632

#vector G
0   0   0   0   0   0   0   0

#vector Ca
0   0   0   0   0   0   0   0

#vector IP3
0   0   0   0   0   0   0   0

#vector z
0   0   0   0   0   0   0   0

""")

f.close()
