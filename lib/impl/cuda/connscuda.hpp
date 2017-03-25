/** @addtogroup Conns
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorConnsImplCudaHPPNDN2017_
#define _NeuroglialNetworkSimulatorConnsImplCudaHPPNDN2017_

#include <valarray>
#include "../connsimpl.hpp"

namespace NNSimulator {

    //! Реализации методов изменения состояний синапсов на графическом процессоре.
    template<class T> class ConnsImplCuda : public ConnsImpl<T> {
    public:

        //! Реализация изменения состояния синапсов, реализованных в классе ConnsSpec. 
        virtual void performStepTimeSpec(
                const T & dt, 
                const T & paramSpec,
                const std::valarray<T> & V,
                T & t,
                std::valarray<T> & I,
                std::valarray<T> & weights
        ) final ;
    };

}

#endif

/*@}*/
