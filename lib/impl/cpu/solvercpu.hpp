/** @addtogroup Solvers
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorSolverImplCPUNDN2017_
#define _NeuroglialNetworkSimulatorSolverImplCPUNDN2017_

#include <valarray>
#include <memory>
#include "../solverimpl.hpp"

namespace NNSimulator {

    //! Класс решателя для универсальных процессоров. 
    template<class T> class SolverImplCPU : public SolverImpl<T> {

    public:

        //! Рассчитать динамику сети.
        virtual void solveExplicitEuler(
            std::unique_ptr<NNSimulator::Neurs<T>> & neurs,
            std::unique_ptr<NNSimulator::Conns<T>> & conns,
            const T & dt,
            const T & simulationTime 
        ) final;

    };

}

#endif

/*@}*/
