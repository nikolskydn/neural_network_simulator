/** @addtogroup SolversImpl
 * @{*/

/** @file */

#ifndef _NNetworkSimulatorSolverImplNDN2017_
#define _NNetworkSimulatorSolverImplNDN2017_

#include <valarray>
#include <memory>

namespace NNSimulator {

    //! Определяет интерфейс реализаций решателей.
    template<class T> class SolverImpl {
        public:

            //! Интерфейс метода, выполняющего расчет динамики сети.
            virtual void solveExplicitEulerSpec(
                const size_t & N,
                const T & VPeak,
                const T & VReset,
                const T &  dt,
                const T & simulationTime,
                const T & neursParamSpec,
                const T & connsParamSpec,
                std::valarray<T> & V,
                std::valarray<bool> & mask,
                std::valarray<T> & I,
                std::valarray<T> & weights,
                T & t
            ) = 0;
    };
}

#endif

/*@}*/

