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

        //! Вспомогательный единичный вектор.
        std::valarray<T> ones;

        //! Вспомогательная обратная маска.
        std::valarray<bool> maskInv;

    public:

        //! Инициализация перед итерациями performStepTimeSpec()
        virtual void performStepTimeSpecInit( const size_t N ) final
        {
            ones.resize( N, static_cast<T>(1) );
        } 

        //! Реализация изменения состояния для модели нейронов, реализованных в классе ConnsSpec. 
        virtual void performStepTimeSpec(
                const T & dt, 
                const T & paramSpec,
                const std::valarray<T> & V,
                const std::valarray<bool> & mask,
                T & t,
                std::valarray<T> & I 
        ) final;

        //! Финализация после итераций performStepTimeSpec()
        virtual void performStepTimeSpecFinalize() final {}
    
    };

}

#endif

/*@}*/
