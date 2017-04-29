echo """
#ifndef _NNetworkSimulatorSettingNDN2017_
#define _NNetworkSimulatorSettingNDN2017_
    // uncomment for build Cuda impl.
    #define NN_CUDA_IMPL 
    // uncomment for testing
    //#define NN_TEST_SOLVERS
    // uncomment for time measurements
    #define TimeDebug 
#endif
""" > ../lib/setting.h
make -C ../lib/impl/cuda
str=`echo $LD_LIBRARY_PATH | grep '../lib/impl/cuda'`
if [ -z "$str" ] ; then 
    source ./export_libs.sh; 
fi
make -f MakefileCuda clean && make -f MakefileCuda && ./nnsimulatorcuda $1
