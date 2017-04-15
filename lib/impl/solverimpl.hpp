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

            /*! \brief Интерфейс реализации модели Е.М. Ижикевича (2003) методом Эйлера.
             *
             * \details Численная схема:
             * 
             *
             * \f$ V_{j+ \frac 1 2} = V_j +  \frac{\Delta t}{2} \left( 0.04V_j^2+5V_j+140-U_j+I_j \right) \f$, 
             *
             * \f$V_{j+1} = V_{j+\frac 1 2} +  
             * \frac{\Delta t}{2} \left( 0.04V_{j+\frac 1 2}^2+5V_{j+\frac 1 2}+140-U_j+I_j \right) \f$,
             *
             * \f$U_{j+1}=a(bV_{j+1}-U_j)\f$,
             *
             * \f${\rm if} \; V\leq 30 \; mV, \; {\rm then} \; \{ V=V_{Reset} \; and \; U+=U_{Reset} \}.\f$
             *
             * Izhikevich E.M. Simple model of spiking neurons// IEEE transactions of neural networks. V.14. N6. 2003. PP.1569--1572. (http://www.izhikevich.org/publications/spikes.pdf)
             */
            virtual void solvePCNN2003E(
                const size_t & nNeurs,
                const size_t & nNeursExc,
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

