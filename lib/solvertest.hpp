/** @addtogroup TestSolvers
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorSolverForTestNDN2017_
#define _NeuroglialNetworkSimulatorSolverForTestNDN2017_

#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <type_traits>
#include "solver.hpp"

namespace NNSimulator {

    template<class T> class Solver;

    //! Класc для тестирования работоспособности проекта.
    template<class T> class SolverForTest: public Solver<T>
    {

            using Solver<T>::nNeurs_;
            using Solver<T>::VNeurs_;
            using Solver<T>::VNeursPeak_;
            using Solver<T>::VNeursReset_;
            using Solver<T>::mNeurs_;
            using Solver<T>::INeurs_;
            using Solver<T>::wConns_;
            using Solver<T>::t_;
            using Solver<T>::st_;
            using Solver<T>::dt_;
            using Solver<T>::pImpl_;

        protected:

            //! Некоторый специальный параметр нейронов.
            T neursParamSpec_;

            //! Некоторый специальный параметр синапсов.
            T connsParamSpec_;

        public:

            //! Конструктор.
            explicit SolverForTest() :  Solver<T>()  {}

            //! Деструктор.
            virtual ~SolverForTest() = default;

            //! Копирующий конструктор.
            SolverForTest( const SolverForTest& ) = delete;

            //! Оператор присваивания.
            SolverForTest& operator=( const SolverForTest& ) = delete;

            //! Перемещающий конструктор.
            SolverForTest( const SolverForTest&& ) = delete;

            //! Перемещающий оператор присваивания.
            SolverForTest& operator=( const SolverForTest&& ) = delete;

            //! Выполнить решение \details Выполняется вызов установленной реализации.
            virtual void solve() final
            {
                pImpl_->solveTest
                ( 
                    nNeurs_,
                    VNeursPeak_,
                    VNeursReset_,
                    dt_,
                    st_,
                    neursParamSpec_,
                    connsParamSpec_,
                    VNeurs_,
                    mNeurs_,
                    INeurs_,
                    wConns_,
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
