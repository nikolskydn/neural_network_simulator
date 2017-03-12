#include "connscpu.hpp"

namespace NNSimulator {

        //! Реализация изменения состояния синапса на универсальном процессоре.
        template<class T>  void ConnsImplCPU<T>::performStepTimeSpec(
                const T & dt, 
                const T & paramSpec,
                const std::valarray<T> & V,
                const std::valarray<bool> & mask,
                T & t,
                std::valarray<T> & I 
        )
        {
            maskInv = !mask;
            I[mask] += (paramSpec*dt*ones)[mask]*V[mask];
            I[maskInv] *= (static_cast<T>(0.5)*ones)[maskInv];
            t += dt;
        }

    template class ConnsImplCPU<float>; 

    template class ConnsImplCPU<double>; 
}
