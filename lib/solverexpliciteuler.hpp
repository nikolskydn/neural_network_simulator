/** @addtogroup Solvers
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorSolverExplicitEulerNDN2017_
#define _NeuroglialNetworkSimulatorSolverExplicitEulerNDN2017_


#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <type_traits>
#include "neursspec.hpp"
#include "connsspec.hpp"
#include "solver.hpp"


namespace NNSimulator {

    template<class T> class Solver;

    template<class T> class SolverExplicitEuler: public Solver<T>
    {

        using Solver<T>::pImpl_;
        using Solver<T>::dt_;
        using Solver<T>::neurs_;
        using Solver<T>::conns_;
        using Solver<T>::simulationTime_;

    public:

        //! Конструктор.
        explicit SolverExplicitEuler() :  Solver<T>()  {}

        //! Деструктор.
        virtual ~SolverExplicitEuler() = default;

        //! Копирующий конструктор.
        SolverExplicitEuler( const SolverExplicitEuler& ) = delete;

        //! Оператор присваивания.
        SolverExplicitEuler& operator=( const SolverExplicitEuler& ) = delete;

        //! Перемещающий конструктор.
        SolverExplicitEuler( const SolverExplicitEuler&& ) = delete;

        //! Перемещающий оператор присваивания.
        SolverExplicitEuler& operator=( const SolverExplicitEuler&& ) = delete;

        //! Выполнить решение.
        virtual void solve() final
        {
            pImpl_->solveExplicitEuler( neurs_, conns_, dt_, simulationTime_ );
        }
    };

} // namespace

#endif

/*@}*/
