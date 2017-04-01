/** @addtogroup Solvers
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorSolverNDN2017_
#define _NeuroglialNetworkSimulatorSolverNDN2017_

#include <iostream>
#include <sstream>
#include <valarray>
#include <iomanip>
#include <type_traits>
#include "formatstream.hpp"

#include "solvertest.hpp"
#include "solverpcnn.hpp"

#include "setting.h"
#ifdef NN_CUDA_IMPL 
    #include "./impl/cuda/solvercuda.hpp"
#else 
    #include "./impl/cpu/solvercpu.hpp"
#endif


namespace NNSimulator {

    template<class T> class SolverForTest;
    template<class T> class SolverPCNN;

    template<class T> class SolverImpl;
    template<class T> class SolverImplCPU;
    template<class T> class SolverImplCuda;

    //! Базовый класс для численных решателей.
    template<class T> class Solver
    {
        protected:

            //! Число нейронов. 
            size_t nNeurs_ {0};

            //! Вектор мембранных потенциалов.
            std::valarray<T> VNeurs_;

            //! Предельное значение потенциала.
            T VNeursPeak_;

            //! Значение потенциала после спайка.
            T VNeursReset_;

            //! Маска, хранящая спайки.
            std::valarray<bool> mNeurs_;

            //! Вектор токов для нейров. 
            std::valarray<T> INeurs_;

            //! Матрица весов nNeurs_ x nNeurs_.
            std::valarray<T> wConns_;

            //! Модельное время. 
            T t_ ;

            //! Шаг по времени.
            T  dt_ ; 

            //! Время симуляции.
            T st_; 

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
                SolverForTestId = 0, //!< явный метод Эйлера для модели некоторой тестовой модели Spec
                SolverPCNNId = 1, //!< модель Е.М. Ижикевича 2003
            };

            //! Фабричный метод создания конкретного решателя. 
            static std::unique_ptr<Solver<T>> createItem( ChildId id )
            {
                std::unique_ptr<Solver<T>> ptr;
                switch( id )
                {
                    case SolverForTestId:
                        ptr = std::unique_ptr<Solver<T>>( std::make_unique<SolverForTest<T>>() );
                    break;
                    case SolverPCNNId:
                        ptr = std::unique_ptr<Solver<T>>( std::make_unique<SolverPCNN<T>>() );
                    break;
                }
                return ptr;
            }
            
            //! Потоковое чтение данных.
            virtual std::istream& read( std::istream& istr ) 
            {
                // solver
                istr >> t_;
                istr >> st_;
                istr >> dt_;
                // neurs
                istr >> nNeurs_ ;
                VNeurs_.resize(nNeurs_);
                mNeurs_.resize(nNeurs_);
                for( auto & e: VNeurs_ ) istr >> e;
                for( auto & e: mNeurs_ ) istr >> e;
                istr >> VNeursPeak_ >> VNeursReset_;
                // conns
                INeurs_.resize(nNeurs_);
                for( auto & e: INeurs_ ) istr >> e;
                size_t nConns = nNeurs_ * nNeurs_;
                wConns_.resize(nConns);
                for( auto & e: wConns_ ) istr >> e;
                return istr;
            }

            //! Потоковая запись данных.
            virtual std::ostream& write( std::ostream& ostr ) const 
            { 
                FormatStream oFStr( ostr );
                // solver
                oFStr << t_;
                oFStr << st_ ;
                oFStr << dt_ ;
                // neurs
                oFStr << nNeurs_;
                for( const auto & e: VNeurs_ ) oFStr << e ;  
                for( const auto & e: mNeurs_ ) oFStr << e ;  
                oFStr << VNeursPeak_ <<  VNeursReset_ ;
                // conns
                for( const auto & e: INeurs_ ) oFStr << e ;
                for( const auto & e: wConns_ ) oFStr << e  ;
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

