/** @addtogroup TestSolver
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

    //! \~russian \brief Класc для тестирования работоспособности проекта. \details Сделан, чтобы можно было вручную проверить числа, которые выдает программа.
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
            using Solver<T>::tEnd_;
            using Solver<T>::dt_;
            using Solver<T>::pImpl_;

        protected:

            //! \~russian \brief Некоторый специальный параметр нейронов.
            T neursParamSpec_;

            //! \~russian \brief Некоторый специальный параметр синапсов.
            T connsParamSpec_;

            /*! \~russian
             * \brief Выполнить решение.
             * \details Выполняется вызов установленной реализации.
             * \param cte текущее модельное время.
             */
            virtual void solveImpl( const T & cte ) final
            {
                pImpl_->solveTest
                ( 
                    nNeurs_,
                    VNeursPeak_,
                    VNeursReset_,
                    dt_,
                    cte,
                    neursParamSpec_,
                    connsParamSpec_,
                    VNeurs_,
                    mNeurs_,
                    INeurs_,
                    wConns_,
                    t_
                );
            }

        public:

            //! \~russian \brief Конструктор.
            explicit SolverForTest() :  Solver<T>()  {}

            //! \~russian \brief Деструктор по умолчанию.
            virtual ~SolverForTest() = default;

            //! \~russian \brief Удаленный копирующий конструктор.
            SolverForTest( const SolverForTest& ) = delete;

            //! \~russian \brief Удаленный оператор присваивания.
            SolverForTest& operator=( const SolverForTest& ) = delete;

            //! \~russian \brief Удаленный перемещающий конструктор.
            SolverForTest( const SolverForTest&& ) = delete;

            //! \~russian \brief Удаленный перемещающий оператор присваивания.
            SolverForTest& operator=( const SolverForTest&& ) = delete;

            /*! \~russian
             * \brief Метод вывода параметров в поток.
             * \details Выводит специальные параметры нейронов и связей.
             * \param ostr ссылка на поток вывода, куда печатаются данные симулятора.
             * \return ссылку на поток вывода, куда были напечатанны данные симулятора.
             */
            virtual std::ostream& write( std::ostream& ostr ) const final
            {
                FormatStream oFStr( Solver<T>::write( ostr ) );
                oFStr << neursParamSpec_ ;
                oFStr << connsParamSpec_ ;
                return oFStr;
            }

            /*! \~russian
             * \brief Метод ввода параметров из потока.
             * \details Заполняет специальные параметры нейронов и связей.
             * \param istr ссылка на поток ввода, откуда берутся данные.
             * \return ссылку на поток ввода, откуда были взяты данные.
             */
            virtual std::istream& read( std::istream& istr ) final
            {
                return ( Solver<T>::read(istr) >> neursParamSpec_ >> connsParamSpec_ );
            }
        };
} // namespace

#endif

/*@}*/
