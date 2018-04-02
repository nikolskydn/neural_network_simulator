/** @addtogroup Solvers
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorSolverPulseCoupledNeuralNetworkIzhik2003EulerNDN2017_
#define _NeuroglialNetworkSimulatorSolverPulseCoupledNeuralNetworkIzhik2003EulerNDN2017_

#include <iostream>
#include <sstream>
#include <valarray>
#include <iomanip>
#include <type_traits>
#include "datacreator.hpp"
#include "solver.hpp"

namespace NNSimulator {

    template<class T> class Solver;

    /*! \~russian \brief Использует численную схему для  модели Е.М. Ижикевича (2003),
     * построенную методом Эйлера.
     *
     * Выполняет вызов метода solvePCNNI2003E из класса SolverImpl.
     */
    template<class T> class SolverPCNNI2003E: public Solver<T>
    {
            using Solver<T>::sNum_;
            using Solver<T>::dNum_;
            using Solver<T>::pImpl_;
            using Solver<T>::pData_;
            using Solver<T>::spikes_;
            using Solver<T>::oscillograms_;
            using Solver<T>::isBinaryWrite_;

        protected:

            //! \~russian \brief Выполнить решение \details Выполняется вызов установленной реализации.
            virtual void solveImpl_( const T & cte ) final
            {
                auto pD = static_cast<DataPCNNI2003<T>*>(pData_.get());
                pImpl_->solvePCNNI2003E
                (
                    pD->nNeurs,
                    pD->nNeursExc,
                    pD->VNeursPeak,
                    pD->aNeurs,
                    pD->bNeurs,
                    pD->cNeurs,
                    pD->dNeurs,
                    pD->wConns,
                    pD->dt,
                    cte,
                    pD->VNeurs,
                    pD->UNeurs,
                    pD->mNeurs,
                    pD->INeurs,
                    pD->t,
                    oscillograms_
                );
            }

        public:

            //! \~russian \brief Конструктор.
            explicit SolverPCNNI2003E() :  Solver<T>(1,1) {}

            //! \~russian \brief Деструктор.
            virtual ~SolverPCNNI2003E() = default;

            //! \~russian \brief Копирующий конструктор.
            SolverPCNNI2003E( const SolverPCNNI2003E& ) = delete;

            //! \~russian \brief Оператор присваивания.
            SolverPCNNI2003E& operator=( const SolverPCNNI2003E& ) = delete;

            //! \~russian \brief Перемещающий конструктор.
            SolverPCNNI2003E( const SolverPCNNI2003E&& ) = delete;

            //! \~russian \brief Перемещающий оператор присваивания.
            SolverPCNNI2003E& operator=( SolverPCNNI2003E&& ) = delete;

            //! \~russian \brief Метод вывода параметров в поток.
            virtual std::ostream& write( std::ostream& ostr ) const final
            {
                auto pD = static_cast<DataPCNNI2003<T>*>(pData_.get());

                if ( isBinaryWrite_ ){
                	BinaryBufferOutS binout( ostr );
                	binout << sNum_ << dNum_;
                }
                else{
                    FormatStream oFStr( ostr );
                    oFStr << sNum_;
                    oFStr << dNum_;
                }

                ostr << *pD;
                return ostr;
            }

            //! \~russian \brief Метод ввода параметров из потока.
            virtual std::istream& read( std::istream& istr ) final
            {  
                auto pD = static_cast<DataPCNNI2003<T>*>( pData_.get() );
                pD->read(istr);
                return istr;
            }

            /*! \~russian
             * \brief Функция проверяет корректность данных, которые
             * находятся в потоке istr.
             * \details Данные в потоке проверяются на то, подходят ли они
             * для входного файла симулятора.
             * \param istr ссылка на поток, откуда ведется считывание.
             * \return true, если данные подходят для входного файла симулятора.
             */
            virtual std::pair<bool, std::string> checkFile( std::istream& istr ) const{
            	auto pD = static_cast<DataPCNNI2003<T>*>( pData_.get() );
            	return pD->checkFile( istr );
            }
        };
} // namespace

#endif

/*@}*/
