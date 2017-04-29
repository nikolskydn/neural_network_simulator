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
        virtual void solvePCNNI2003E
        (
            const size_t & nNeurs,
            const size_t & nNeursExc,
            const T & VNeursPeak,
            const std::valarray<T> aNeurs,
            const std::valarray<T> bNeurs,
            const std::valarray<T> cNeurs,
            const std::valarray<T> dNeurs,
            const std::valarray<T> & wConns,
            const T &  dt,
            const T & tEnd,
            std::valarray<T> & VNeurs,
            std::valarray<T> & UNeurs,
            std::valarray<bool> & mNeurs,
            std::valarray<T> & INeurs,
            T & t,
            std::deque<std::pair<T,std::valarray<T>>> & oscillograms
        ) override final;

    };
}

#endif

/*@}*/

