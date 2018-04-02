function MV {
    mv $1 $2
    if [ $? ] ; then 
        echo -e "\033[33;2m    makercpu.sh:    move $1 to $2 ok\033[0m"
    fi
}

echo """
#ifndef _NNetworkSimulatorSettingNDN2017_
#define _NNetworkSimulatorSettingNDN2017_
    // uncomment for build Cuda impl.
    //#define NN_CUDA_IMPL 

    // uncomment for testing
    #define NN_TEST_SOLVERS

    // uncomment for time measurements
    //#define TimeDebug 
#endif
""" > ../lib/setting.h
if [ -n "$1" ] ; then
    export LIBSDIR="$1"
    echo -e "\033[33;2m    makercpu: LIBSDIR=$LIBSDIR\033[0m"
else 
    echo -e "\033[32;1m    makercpu ERROR: set directory for lib\033[0m"
    exit 1
fi
if [ -n "$2" ] ; then
    BINDIR="$2"
    echo -e "\033[33;2m    makercpu: BINDIR=$BINDIR\033[0m"
else 
    echo -e "\033[32;1m    makercpu ERROR: set directory for bin\033[0m"
    exit 1
fi
make -C ../lib/impl/cpu/ clean
make -C ../lib/impl/cpu/ libnnsolvercputest.so
MV ../lib/impl/cpu/libnnsolvercputest.so ${LIBSDIR}
make -f MakefileCPU clean
make -f MakefileCPU
MV test_nnsolvers_cpu ${BINDIR}
MV test_unn270117_funcs_cpu ${BINDIR}

MV solverUNN270117RK.in ${BINDIR}
MV solverUNN270117RK.end ${BINDIR}
