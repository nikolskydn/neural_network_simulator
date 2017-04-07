/** @addtogroup Solvers
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorSolverPulseCoupledNeuralNetworkNDN2017_
#define _NeuroglialNetworkSimulatorSolverPulseCoupledNeuralNetworkNDN2017_

#include <iostream>
#include <sstream>
#include <valarray>
#include <iomanip>
#include <type_traits>
#include "solver.hpp"

namespace NNSimulator {

    template<class T> class Solver;

    /*! \brief Класc содержит реализацию модели Е.М. Ижикевича, принадлежащую классу импульсно-связанных нейронных сетей (pulse-coupled neural networks).
     *
     * В основе модели лежит система двух обыкновенных дифференциальных уравнений:
     *
     * \f$\frac{dV}{dt}=0.04V^2+5V+140-U+I\f$,
     *
     * \f$\frac{dU}{dt}=a(bV-U)\f$,
     *
     * \f${\rm if} \; V\leq 30 \; mV, \; {\rm then} \; \{ V=V_{Reset} \; and \; U+=U_{Reset} \}.\f$
     *
     * Izhikevich E.M. Simple model of spiking neurons// IEEE transactions of neural networks. V.14. N6. 2003. PP.1569--1572. (http://www.izhikevich.org/publications/spikes.pdf)
     */
    template<class T> class SolverPCNN: public Solver<T>
    {

            using Solver<T>::sNum_;
            using Solver<T>::nNeurs_;
            using Solver<T>::nNeursExc_;
            using Solver<T>::VNeurs_;
            using Solver<T>::VNeursPeak_;
            using Solver<T>::VNeursReset_;
            using Solver<T>::mNeurs_;
            using Solver<T>::INeurs_;
            using Solver<T>::wConns_;
            using Solver<T>::t_;
            using Solver<T>::tEnd_;
            using Solver<T>::dt_;
            //using Solver<T>::pOutStream_;
            using Solver<T>::dtDump_;
            using Solver<T>::pImpl_;

        protected:

            //! Вектор мембранной восстановительной переменной \f$U\f$. \details Обеспечивает обратную связь. Определяет активацию ионного тока \f$K^+\f$ и деактивацию ионов \f$Na^+\f$.
            std::valarray<T> UNeurs_ {};
            
            //! Вектор параметров \f$a\f$ из основной системы ОДУ. \details Определяет временной ммасштаб восстановительной переменной \f$U\f$.
            std::valarray<T> aNeurs_ {};

            //! Вектор параметров \f$b\f$ из основной системы ОДУ. \details Определяет чувствительность восстановительной переменной \f$U\f$.
            std::valarray<T> bNeurs_ {};

            //! Вектор для вычисления значений мембранных потенциалов после спайка.
            std::valarray<T> cNeurs_ {};

            //! Вектор для вычисления значений восстановительной переменной \f$U\f$ после спайка.
            std::valarray<T> dNeurs_ {};

            //! Выполнить решение \details Выполняется вызов установленной реализации.
            virtual void solveImpl( const T & cte ) final
            {
                pImpl_->solvePCNN
                (
                    nNeurs_,
                    nNeursExc_,
                    VNeursPeak_,
                    VNeursReset_,
                    aNeurs_,
                    bNeurs_,
                    cNeurs_,
                    dNeurs_,
                    dt_,
                    cte,
                    VNeurs_,
                    UNeurs_,
                    mNeurs_,
                    INeurs_,
                    wConns_,
                    t_
                );
            }

        public:

            //! Конструктор.
            explicit SolverPCNN() :  Solver<T>()  { sNum_ = 1; }

            //! Деструктор.
            virtual ~SolverPCNN() = default;

            //! Копирующий конструктор.
            SolverPCNN( const SolverPCNN& ) = delete;

            //! Оператор присваивания.
            SolverPCNN& operator=( const SolverPCNN& ) = delete;

            //! Перемещающий конструктор.
            SolverPCNN( const SolverPCNN&& ) = delete;

            //! Перемещающий оператор присваивания.
            SolverPCNN& operator=( const SolverPCNN&& ) = delete;


            //! Метод вывода параметров в поток. 
            virtual std::ostream& write( std::ostream& ostr ) const final
            {
                FormatStream oFStr( Solver<T>::write( ostr ) );
                //oFStr << sNum_;
                for( const auto & e: UNeurs_ ) oFStr << e ;  
                for( const auto & e: aNeurs_ ) oFStr << e ;  
                for( const auto & e: bNeurs_ ) oFStr << e ;  
                for( const auto & e: cNeurs_ ) oFStr << e ;  
                for( const auto & e: dNeurs_ ) oFStr << e ;  
                return oFStr;
            }

            //! Метод ввода параметров из потока.  
            virtual std::istream& read( std::istream& istr ) final
            {  
                Solver<T>::read(istr);

                UNeurs_.resize(nNeurs_);
                for( auto & e: UNeurs_ ) istr >> e;
                aNeurs_.resize(nNeurs_);
                for( auto & e: aNeurs_ ) istr >> e;
                bNeurs_.resize(nNeurs_);
                for( auto & e: bNeurs_ ) istr >> e;
                cNeurs_.resize(nNeurs_);
                for( auto & e: cNeurs_ ) istr >> e;
                dNeurs_.resize(nNeurs_);
                for( auto & e: dNeurs_ ) istr >> e;
                return istr;
            }
        };
} // namespace

#endif

/*@}*/
