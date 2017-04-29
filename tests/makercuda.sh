echo """
#ifndef _NNetworkSimulatorSettingNDN2017_
#define _NNetworkSimulatorSettingNDN2017_

    // uncomment for build Cuda impl.
    #define NN_CUDA_IMPL 

    // uncomment for testing
    #define NN_TEST_SOLVERS

    // for time measurements
    //#define TimeDebug 

#endif
""" > ../lib/setting.h
make -C ../lib/impl/cuda
source  export_libs.sh
make -f MakefileCuda clean && make -f MakefileCuda && ./test_solvers_cuda
