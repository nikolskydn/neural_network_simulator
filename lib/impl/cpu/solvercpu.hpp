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

            //! Заполняет входной массив случайными числами из нормалного распределения с использованием ГПСЧ "Вихрь Мерсенна".
            void makeRandn( std::valarray<T> & v );

            //! Вспомогательный единичный вектор.
            std::valarray<T> ones_;

            //! Вспомогательная обратная маска.
            std::valarray<bool> mInv_;

        public:

            //! Рассчитать динамику сети.
            virtual void solveTest(
                const size_t & nNeurs,
                const T & VNeursPeak,
                const T & VNeursReset,
                const T &  dt,
                const T & st,
                const T & neursParamSpec,
                const T & connsParamSpec,
                std::valarray<T> & VNeurs,
                std::valarray<bool> & mNeurs,
                std::valarray<T> & INeurs,
                std::valarray<T> & wConns,
                T & t
            ) final;

            //! Реализация модели Е.М. Ижикевича класса PCNN.
            virtual void solvePCNN(
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
            ) final;
    };
}

#endif

/*@}*/
