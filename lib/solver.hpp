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
#include "formatstream.hpp"

#include "solverexpliciteulerspec.hpp"

#include "setting.h"
#ifdef NN_CUDA_IMPL 
    #include "./impl/cuda/solvercuda.hpp"
#else 
    #include "./impl/cpu/solvercpu.hpp"
#endif


namespace NNSimulator {

    template<class T> class SolverExplicitEulerSpec;

    template<class T> class SolverImpl;
    template<class T> class SolverImplCPU;
    template<class T> class SolverImplCuda;

    //! Базовый класс для численных решателей.
    template<class T> class Solver
    {
        protected:

            //! Число нейронов в моделируемой сети.
            size_t N_ {0};

            //! Вектор мембранных потенциалов.
            std::valarray<T> V_;

            //! Предельное значение потенциала.
            T VPeak_;

            //! Значение потенциала после спайка.
            T VReset_;

            //! Маска, хранящая спайки.
            std::valarray<bool> mask_;

            //! Вектор токов.
            std::valarray<T> I_;

            //! Матрица весов \details С-like style.
            std::valarray<T> weights_;

            //! Модельное время 
            T t_ ;

            //! Шаг по времени.
            T  dt_ ;

            //! Время симуляции
            T simulationTime_;

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
                SolverExplicitEulerSpecId = 0, //!< явный метод Эйлера для модели некоторой тестовой модели Spec
                SolverExplicitRungeKutta45 = 1,
            };

            //! Фабричный метод создания конкретного решателя. 
            static std::unique_ptr<Solver<T>> createItem( ChildId id )
            {
                std::unique_ptr<Solver<T>> ptr;
                switch( id )
                {
                    case SolverExplicitEulerSpecId:
                        ptr = std::unique_ptr<Solver<T>>( std::make_unique<SolverExplicitEulerSpec<T>>() );
                    break;
                }
                return ptr;
            }
            
            //! Потоковое чтение данных.
            virtual std::istream& read( std::istream& istr ) 
            {
                // solver
                istr >> t_;
                istr >> simulationTime_;
                istr >> dt_;
                // neurs
                istr >> N_ ;
                V_.resize(N_);
                mask_.resize(N_);
                for( auto & e: V_ ) istr >> e;
                for( auto & e: mask_ ) istr >> e;
                istr >> VPeak_ >> VReset_;
                // conns
                I_.resize(N_);
                for( auto & e: I_ ) istr >> e;
                weights_.resize(N_*N_);
                for( auto & e: weights_ ) istr >> e;
                return istr;
            }

            //! Потоковая запись данных.
            virtual std::ostream& write( std::ostream& ostr ) const 
            { 
                FormatStream oFStr( ostr );
                // solver
                oFStr << t_;
                oFStr << simulationTime_ ;
                oFStr << dt_ ;
                // neurs
                oFStr << N_;
                for( const auto & e: V_ ) oFStr << e ;  
                for( const auto & e: mask_ ) oFStr << e ;  
                oFStr << VPeak_ <<  VReset_ ;
                // conns
                for( const auto & e: I_ ) 
                    oFStr << e ;
                for( const auto & e: weights_ ) 
                    oFStr << e  ;
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

