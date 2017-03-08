#include "neurscpu.hpp"
#include <iostream>
namespace NNSimulator {

        template<class T>
        void NeursImplCPU<T>::performStepTimeSpec(
                const T & dt, 
                const std::valarray<T> & I,
                const T & VPeak,
                const T & VReset,
                const T & paramSpec,
                T & t,
                std::valarray<T> & V, 
                std::valarray<bool> & mask
        )
        {
            std::valarray<bool> maskInv = !mask;
            std::valarray<T> ones(1,I.size());
            V[maskInv] += (paramSpec*dt*ones)*I[maskInv];
            V[mask] = VReset;
            t += dt;
            mask = V>VPeak;
        }

    template class NeursImplCPU<float>; 

    template class NeursImplCPU<double>; 
}

