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
            virtual void solveTest(
                const size_t & nNeurs,
                const T & VNeursPeak,
                const T & VNeursReset,
                const T &  dt,
                const T & st,
                const T & neursParamSpec,
                const T & connsParamSpec,
                std::valarray<T> & VNeurs,
                std::valarray<bool> & mNeurs,
                std::valarray<T> & INeurs,
                std::valarray<T> & wConns,
                T & t
            ) = 0;

            //! Интерфейс реализации модели Е.М. Ижикевича, принадлежащей классу сетей PCNN.
            virtual void solvePCNN(
                const size_t & nNeurs,
                const T & VNeursPeak,
                const T & VNeursReset,
                const std::valarray<T> aNeurs,
                const std::valarray<T> bNeurs,
                const std::valarray<T> cNeurs,
                const std::valarray<T> dNeurs,
                const T &  dt,
                const T & st,
                std::valarray<T> & VNeurs,
                std::valarray<T> & UNeurs_,
                std::valarray<bool> & mNeurs,
                std::valarray<T> & INeurs,
                std::valarray<T> & wConns,
                T & t
            ) = 0;
    };
}
#endif

/*@}*/

