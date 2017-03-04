/** @addtogroup Neurs
 * @{*/

/** @file */

#ifndef _NNetworkSimulatorNeursImplCudaHPPNDN2017_
#define _NNetworkSimulatorNeursImplCudaHPPNDN2017_

#include <valarray>
#include "../neursimpl.hpp"

namespace NNSimulator {

    //! Реализации методов изменения состояний нейронов на графическом процессоре.
    template<class T> class NeursImplCuda : public NeursImpl<T> {
    public:

        //! Реализация изменения состояния для модели нейронов, реализованных в классе NeursSpec. 
        virtual void performStepTimeSpec(
                const T & dt, 
                const T & paramSpec,
                const std::valarray<T> & I,
                T & t,
                std::valarray<T> & V 
        ) final ;
    };

}

#endif

/*@}*/

