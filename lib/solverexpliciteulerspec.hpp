/** @addtogroup Solvers
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorSolverExplicitEulerSpecNDN2017_
#define _NeuroglialNetworkSimulatorSolverExplicitEulerSpecNDN2017_

#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <type_traits>
#include "solver.hpp"

namespace NNSimulator {

    template<class T> class Solver;

    template<class T> class SolverExplicitEulerSpec: public Solver<T>
    {

            using Solver<T>::N_;
            using Solver<T>::V_;
            using Solver<T>::VPeak_;
            using Solver<T>::VReset_;
            using Solver<T>::mask_;
            using Solver<T>::I_;
            using Solver<T>::weights_;
            using Solver<T>::t_;
            using Solver<T>::simulationTime_;
            using Solver<T>::dt_;
            using Solver<T>::pImpl_;

        protected:

            //! Некоторый специальный параметр нейронов.
            T neursParamSpec_;

            //! Некоторый специальный параметр синапсов.
            T connsParamSpec_;

        public:

            //! Конструктор.
            explicit SolverExplicitEulerSpec() :  Solver<T>()  {}

            //! Деструктор.
            virtual ~SolverExplicitEulerSpec() = default;

            //! Копирующий конструктор.
            SolverExplicitEulerSpec( const SolverExplicitEulerSpec& ) = delete;

            //! Оператор присваивания.
            SolverExplicitEulerSpec& operator=( const SolverExplicitEulerSpec& ) = delete;

            //! Перемещающий конструктор.
            SolverExplicitEulerSpec( const SolverExplicitEulerSpec&& ) = delete;

            //! Перемещающий оператор присваивания.
            SolverExplicitEulerSpec& operator=( const SolverExplicitEulerSpec&& ) = delete;

            //! Выполнить решение.
            virtual void solve() final
            {
                pImpl_->solveExplicitEulerSpec
                ( 
                    N_,
                    VPeak_,
                    VReset_,
                    dt_,
                    simulationTime_,
                    neursParamSpec_,
                    connsParamSpec_,
                    V_,
                    mask_,
                    I_,
                    weights_,
                    t_
                );
            }

            //! Метод вывода параметров в поток. 
            virtual std::ostream& write( std::ostream& ostr ) const final
            {
                FormatStream oFStr( Solver<T>::write( ostr ) );
                oFStr << neursParamSpec_ ;
                oFStr << connsParamSpec_ ;
                return oFStr;
            }

            //! Метод ввода параметров из потока.  
            virtual std::istream& read( std::istream& istr ) final
            {
                return ( Solver<T>::read(istr) >> neursParamSpec_ >> connsParamSpec_ );
            }
        };
} // namespace

#endif

/*@}*/
