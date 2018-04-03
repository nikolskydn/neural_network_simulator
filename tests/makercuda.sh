function MV {
    mv $1 $2
    if [ $? ] ; then 
        echo -e "\033[33;2m    makercuda.sh:    move $1  to $2 ok\033[0m"
    fi
}

function CP {
	cp $1 $2
	if [ $? ] ; then
		echo -e "\033[33;2m    makercuda.sh:    copy $1 to $2 ok\033[0m"
	fi
}

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
if [ -n "$1" ] ; then
    export LIBSDIR="$1"
    echo -e "\033[33;2m    makercuda: LIBSDIR=$LIBSDIR\033[0m"
else 
    echo -e "\033[32;1m    makercuda ERROR: set directory for lib\033[0m"
    exit 1
fi
if [ -n "$2" ] ; then
    BINDIR="$2"
    echo -e "\033[33;2m    makercuda: BINDIR=$BINDIR\033[0m"
else 
    echo -e "\033[32;1m    makercuda ERROR: set directory for bin\033[0m"
    exit 1
fi
make -C ../lib/impl/cuda clean
make -C ../lib/impl/cuda libnnsolvercudatest.so
MV ../lib/impl/cuda/libnnsolvercudatest.so ${LIBSDIR}
make -f MakefileCuda clean
make -f MakefileCuda 
MV test_nnsolvers_cuda ${BINDIR}

CP solverpcnni2003e.in ${BINDIR}
CP solverpcnni2003e.end ${BINDIR}
