/** @addtogroup Solvers
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorSolverNDN2017_
#define _NeuroglialNetworkSimulatorSolverNDN2017_


#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <type_traits>
#include "neursspec.hpp"
#include "connsspec.hpp"
#include "formatstream.hpp"

// children:
#include "../lib/solverexpliciteuler.hpp"

#include "setting.h"
#ifdef NN_CUDA_IMPL 
    #include "./impl/cuda/solvercuda.hpp"
#else 
    #include "./impl/cpu/solvercpu.hpp"
#endif


namespace NNSimulator {

    template<class T> class SolverExplicitEuler;

    template<class T> class SolverImpl;
    template<class T> class SolverImplCPU;
    template<class T> class SolverImplCuda;

    template<class T> class Conns;
    template<class T> class Neurs;

    //! Базовый класс для численных решателей. \details Хранит основные данные, необходимые для решателей: объекты типа Neurs (нейроны), Conns (синапсы), шаг дискретизации. Определяет интерфейс метода, выполняющего расчеты (solve()).
    template<class T> class Solver
    {
    protected:

        //! Группа нейронов.
        std::unique_ptr<typename NNSimulator::Neurs<T>> neurs_;

        //! Группа синапсов.
        std::unique_ptr<typename NNSimulator::Conns<T>> conns_;

        //! Шаг по времени.
        T  dt_ ;

        //! Время симуляции
        T simulationTime_;

        //! Идентификатор нейрона.
        typename NNSimulator::Neurs<T>::ChildId neursId_;

        //! Идентификатор синапса.
        typename NNSimulator::Conns<T>::ChildId connsId_;

        //! Указатель на реализацию. 
        std::unique_ptr<SolverImpl<T>> pImpl_;


    public:

            //! Конструктор.
            explicit Solver() :  pImpl_( 
                #ifdef NN_CUDA_IMPL
                    std::make_unique<SolverImplCuda<T>>() 
                #else 
                    std::make_unique<SolverImplCPU<T>>() 
                #endif
            ) { }

            //! Деструктор.
            virtual ~Solver() = default;

            //! Копирующий конструктор.
            Solver( const Solver& ) = delete;

            //! Оператор присваивания.
            Solver& operator=( const Solver& ) = delete;

            //! Перемещающий конструктор.
            Solver( const Solver&& ) = delete;

            //! Перемещающий оператор присваивания.
            Solver& operator=( const Solver&& ) = delete;

            //! Перечисление с типами решателей.
            enum ChildId : size_t
            {  
                SolverExplicitEulerId = 0, //!< явный метод Эйлера
                SolverExplicitRungeKutta45 = 1,
            };

            //! Фабричный метод создания конкретного решателя. 
            static std::unique_ptr<Solver<T>> createItem( ChildId id )
            {
                std::unique_ptr<Solver<T>> ptr;
                switch( id )
                {
                    case SolverExplicitEulerId:
                        ptr = std::unique_ptr<Solver<T>>( std::make_unique<SolverExplicitEuler<T>>() );
                    break;
                }
                return ptr;
            }
            
            //! Потоковое чтение данных.
            virtual std::istream& read( std::istream& istr ) 
            {
                istr >> simulationTime_;
                istr >> dt_;
                size_t tmpId;
                istr >> tmpId;
                neursId_ = static_cast<typename NNSimulator::Neurs<T>::ChildId>( tmpId );
                neurs_ = NNSimulator::Neurs<T>::createItem( neursId_ ); 
                istr >> *neurs_;
                istr >> tmpId;
                connsId_ = static_cast<typename NNSimulator::Conns<T>::ChildId>(tmpId);
                conns_ = conns_->createItem( connsId_ );     
                istr >> *conns_;
                neurs_->setCurrents( conns_->getCurrents() );
                conns_->setPotentials( neurs_->getPotentials() ); 
                conns_->setMasks( neurs_->getMasks() ); 
                return istr;
            }

            //! Потоковая запись данных.
            virtual std::ostream& write( std::ostream& ostr ) const 
            { 
                FormatStream oFStr( ostr );
                oFStr << simulationTime_ ;
                oFStr << dt_ ;
                oFStr << neursId_  << *neurs_ ;
                oFStr << connsId_  << *conns_ ;
                return oFStr;
            }
    
            //! Выполнить решение.
            virtual void solve() = 0; 
    };

} // namespace


//! Оператор потокового вывода.
template<class T>
std::ostream& operator<<( std::ostream & ostr, const NNSimulator::Solver<T> & item)
{
    return (item.write(ostr));
}

//! Оператор потокова ввода.
template<class T>
std::istream& operator>>( std::istream & istr, NNSimulator::Solver<T> & item)
{
    return (item.read(istr));
}

#endif

/*@}*/

