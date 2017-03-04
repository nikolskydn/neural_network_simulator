/** @addtogroup Neurs
 * @{*/

/** @file */

#ifndef _NNetworkSimulatorNeursImplNDN2017_
#define _NNetworkSimulatorNeursImplNDN2017_

#include <valarray>

namespace NNSimulator {

    //! Определяет интерфейс реализаций методов изменения состояний нейронов performStepTime*() за некоторый промежуток времени dt.
    template<class T> class NeursImpl {
    public:

        //! Интерфейс метода для реализаций изменения состояния нейронов NeursSpec. \details Изменение потенциала \f$V\f$ за промежуток времени \f$dt\f$определяется следующим выражением \f$V=V+specParam\cdot I\cdot dt\f$, где \f$I\f$ --- ток, \f$specParam\f$ --- некоторый параметр.  
        virtual void performStepTimeSpec(
                const T & dt, 
                const T & paramSpec,
                const std::valarray<T> & I,
                T & t,
                std::valarray<T> & V 
        ) = 0;

    };
}

#endif

/*@}*/

