#include "solvercpu.hpp"
#include <memory>

namespace NNSimulator {

        //! Реализация решателя на универсальных процессорах.
        template<class T>  void SolverImplCPU<T>::solveExplicitEulerSpec(
            const size_t & N,
            const T & VPeak,
            const T & VReset,
            const T &  dt,
            const T & simulationTime,
            const T & neursParamSpec,
            const T & connsParamSpec,
            std::valarray<T> & V,
            std::valarray<bool> & mask,
            std::valarray<T> & I,
            std::valarray<T> & weights,
            T & t
        )
        {
            ones.resize( N, static_cast<T>(1) );
            maskInv.resize( N, false ); 
            maskInv = !mask;

            V[maskInv] += (neursParamSpec*dt*ones)[maskInv]*static_cast<std::valarray<T>>(I[maskInv]);
            V[mask] = VReset;
            mask = V>VPeak;
            maskInv = !mask;
            I[mask] += (connsParamSpec*dt*ones)[mask]*static_cast<std::valarray<T>>(V[mask]);
            I[maskInv] *= (static_cast<T>(0.5)*ones)[maskInv];
            t += dt;
        }

    template class SolverImplCPU<float>; 

    template class SolverImplCPU<double>; 
}
