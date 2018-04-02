#include "solvercpu.hpp"

#include <memory>
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <valarray>
#include <random>

#include "../../setting.h"

#define TimeDebug

#ifndef DEBUG
	#define DEBUG 0
#else
	#undef DEBUG
	#define DEBUG 0
#endif

#if DEBUG >= 1
    #include <fstream>
    #include <string>
    #include <sstream>
#endif

#include "solvercpu-pcnni2003e.icc"
#include "solvercpu-unn270117.icc"

namespace NNSimulator {

    template class SolverImplCPU<float>; 
    template class SolverImplCPU<double>;
    template class SolverImplCPU<long double>;

}
