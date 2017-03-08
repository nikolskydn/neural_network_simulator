/** @addtogroup Conns
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorConnsImplNDN2017_
#define _NeuroglialNetworkSimulatorConnsImplNDN2017_

#include <valarray>

namespace NNSimulator {

    //! Определяет интерфейс реализаций методов изменения состояний нейронов performStepTime*() за некоторый промежуток времени dt.
    template<class T> class ConnsImpl {
    public:

        //! Интерфейс метода для реализаций изменения состояния нейронов ConnsSpec. \details Изменение потенциала определяется следующим выражением \f$V=V+specParam\cdot I\cdot dt\f$. 
        virtual void performStepTimeSpec(
                const T & dt, 
                const T & paramSpec,
                const std::valarray<T> & V,
                T & t,
                std::valarray<T> & I 
        ) = 0;

    };

}

#endif

/*@}*/
