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

            //! Вспомогательный единичный вектор.
            std::valarray<T> ones;

            //! Вспомогательная обратная маска.
            std::valarray<bool> maskInv;

        public:

            //! Рассчитать динамику сети.
            virtual void solveExplicitEulerSpec(
                const size_t & N_,
                const T & VPeak_,
                const T & VReset_,
                const T &  dt_,
                const T & simulationTime_,
                const T & neursParamSpec_,
                const T & connsParamSpec_,
                std::valarray<T> & V_,
                std::valarray<bool> & mask_,
                std::valarray<T> & I_,
                std::valarray<T> & weights_,
                T & t_
            ) final;
    };

}

#endif

/*@}*/
