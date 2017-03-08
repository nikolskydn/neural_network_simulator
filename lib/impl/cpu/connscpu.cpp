#include "connscpu.hpp"

namespace NNSimulator {

        //! Реализация изменения состояния синапса на универсальном процессоре.
        template<class T>  void ConnsImplCPU<T>::performStepTimeSpec(
                const T & dt, 
                const T & paramSpec,
                const std::valarray<T> & V,
                T & t,
                std::valarray<T> & I 
        )
        {
            // bad solution
            for(int i=0; i<V.size(); ++i )
            {
                if(V[i]>=10) I[i] += paramSpec*V[i]*dt;
                else I[i] *= 0.5;
            }
            t += dt;
        }

    template class ConnsImplCPU<float>; 

    template class ConnsImplCPU<double>; 
}
