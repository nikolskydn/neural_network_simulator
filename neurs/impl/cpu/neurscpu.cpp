#include "neurscpu.hpp"

namespace NNSimulator {

        template<class T>
        void NeursImplCPU<T>::performStepTimeSpec(
                const T & dt, 
                const T & paramSpec,
                const std::valarray<T> & I,
                T & t,
                std::valarray<T> & V 
        )
        {
            V += paramSpec*I*dt;
            t += dt;
        }

    template class NeursImplCPU<float>; 

    template class NeursImplCPU<double>; 
}

