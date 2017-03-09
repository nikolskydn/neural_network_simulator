/** @addtogroup Conns
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorConnsImplCudaHPPNDN2017_
#define _NeuroglialNetworkSimulatorConnsImplCudaHPPNDN2017_

#include <valarray>
#include "../connsimpl.hpp"

namespace NNSimulator {

    //! Реализации методов изменения состояний нейронов на графическом процессоре.
    template<class T> class ConnsImplCuda : public ConnsImpl<T> {
    public:

        //! Реализация изменения состояния для модели нейронов, реализованных в классе ConnsSpec. 
        virtual void performStepTimeSpec(
                const T & dt, 
                const T & paramSpec,
                const std::valarray<T> & V,
                T & t,
                std::valarray<T> & I
        ) final ;
    };

}

#endif

/*@}*/
