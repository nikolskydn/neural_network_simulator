/** @addtogroup Solvers
 * @{*/

/** @file */

#ifndef _NNetworkSimulatorSolverImplNDN2017_
#define _NNetworkSimulatorSolverImplNDN2017_

#include <valarray>
#include <memory>

#include "../neurs.hpp"
#include "../conns.hpp"

namespace NNSimulator {

    //! Определяет интерфейс реализаций решателей.
    template<class T> class SolverImpl {
    public:

        //! Интерфейс метода, выполняющего расчет динамики сети.
        virtual void solveExplicitEuler(
                // neurs
            //const T & dt, 
            //const std::valarray<T> & I,
            //const T & VPeak,
            //const T & VReset,
            //const T & paramSpec,
            //T & t,
            //std::valarray<T> & V, 
            //std::valarray<bool> & mask
                // conns
            //const T & dt,
            //const T & paramSpec,
            //const std::valarray<T> & V,
            //const std::valarray<bool> & mask,
            //T & t,
            //std::valarray<T> & I,
            //std::valarray<T> & weights
            std::unique_ptr<NNSimulator::Neurs<T>> & neurs,
            std::unique_ptr<NNSimulator::Conns<T>> & conns,
            const T  & dt, 
            const T & simulationTime 
        ) = 0;

    };
}

#endif

/*@}*/

