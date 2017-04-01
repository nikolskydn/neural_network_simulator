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
        virtual void solveTest(
            const size_t & nNeurs,
            const T & VPeak,
            const T & VReset,
            const T &  dt,
            const T & simulationTime,
            const T & neursParamSpec,
            const T & connsParamSpec,
            std::valarray<T> & VNeurs,
            std::valarray<bool> & mNeurs,
            std::valarray<T> & INeurs,
            std::valarray<T> & wConns,
            T & t
        ) final;

        //! Реализация модели Е.М. Ижикевича класса PCNN.
        virtual void solvePCNN(
            const size_t & nNeurs,
            const T & VNeursPeak,
            const T & VNeursReset,
            const std::valarray<T> aNeurs,
            const std::valarray<T> bNeurs,
            const std::valarray<T> cNeurs,
            const T &  dt,
            const T & st,
            std::valarray<T> & VNeurs,
            std::valarray<T> & UNeurs_,
            std::valarray<bool> & mNeurs,
            std::valarray<T> & INeurs,
            std::valarray<T> & wConns,
            T & t
        ) final {};

    };
}

#endif

/*@}*/

