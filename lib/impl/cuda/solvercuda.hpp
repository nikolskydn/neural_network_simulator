/** @addtogroup Solver
 * @{*/

/** @file */

#ifndef _NNetworkSimulatorSolverImplCudaHPPNDN2017_
#define _NNetworkSimulatorSolverImplCudaHPPNDN2017_

#include <valarray>
#include <memory>
#include "../solverimpl.hpp"

namespace NNSimulator {

    //! Реализации решателей на графическом процессоре.
    template<class T> class SolverImplCuda : public SolverImpl<T> {
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

