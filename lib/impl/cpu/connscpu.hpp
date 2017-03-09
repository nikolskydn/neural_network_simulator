/** @addtogroup Conns
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorConnsImplCPUNDN2017_
#define _NeuroglialNetworkSimulatorConnsImplCPUNDN2017_

#include <valarray>
#include "../connsimpl.hpp"

namespace NNSimulator {

    //! Реализации методов изменения состояний нейронов на универсальном процессоре.
    template<class T> class ConnsImplCPU : public ConnsImpl<T> {
    public:
        //! Реализация изменения состояния для модели нейронов, реализованных в классе ConnsSpec. 
        virtual void performStepTimeSpec(
                const T & dt, 
                const T & paramSpec,
                const std::valarray<T> & V,
                T & t,
                std::valarray<T> & I 
        ) final;
    
    };

}

#endif

/*@}*/
