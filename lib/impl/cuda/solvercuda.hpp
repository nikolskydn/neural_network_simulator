/** @addtogroup SolversImpl
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

        //! Реализация модели Е.М. Ижикевича (2003).
        virtual void solvePCNNI2003E(
            const size_t & nNeurs,
            const size_t & nNeursExc,
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

