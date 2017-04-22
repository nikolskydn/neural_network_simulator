echo """
#ifndef _NNetworkSimulatorSettingNDN2017_
#define _NNetworkSimulatorSettingNDN2017_

    // uncomment for build Cuda impl.
    //#define NN_CUDA_IMPL 

    // uncomment for testing
    #define NN_TEST_SOLVERS

#endif
""" > ../lib/setting.h
make -C ../lib/impl/cpu/
source  export_libs.sh
make -f MakefileCPU clean && make -f MakefileCPU && ./test_solvers_cpu
