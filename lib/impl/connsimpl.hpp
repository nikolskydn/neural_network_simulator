/** @addtogroup Conns
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorConnsImplNDN2017_
#define _NeuroglialNetworkSimulatorConnsImplNDN2017_

#include <valarray>
#include <memory>

namespace NNSimulator {

    //! Определяет интерфейс реализаций методов изменения состояний нейронов performStepTime*() за некоторый промежуток времени dt.
    template<class T> class ConnsImpl {
    public:

        //! Инициализация перед performStepTime()
        virtual void performStepTimeSpecInit( const size_t N ) = 0;

        //! Интерфейс метода для реализаций изменения состояния нейронов ConnsSpec. \details Изменение потенциала определяется следующим выражением \f$V=V+specParam\cdot I\cdot dt\f$. 
        virtual void performStepTimeSpec(
                const T & dt,
                const T & paramSpec,
                const std::valarray<T> & V,
                const std::valarray<bool> & mask,
                T & t,
                std::valarray<T> & I,
                std::valarray<T> & weights
        ) = 0;

        //! Финализация после итераций performStepTimeSpec()
        virtual void performStepTimeSpecFinalize() = 0;
    };

}

#endif

/*@}*/
