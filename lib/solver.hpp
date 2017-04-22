/** @addtogroup Solvers
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorSolverNDN2017_
#define _NeuroglialNetworkSimulatorSolverNDN2017_

#include <iostream>
#include <sstream>
#include <valarray>
#include <vector>
#include <iomanip>
#include <type_traits>
#include "formatstream.hpp"

#include "data.hpp"
#include "solverpcnni2003e.hpp"

#include "setting.h"
#ifdef NN_CUDA_IMPL 
    #include "./impl/cuda/solvercuda.hpp"
#else 
    #include "./impl/cpu/solvercpu.hpp"
#endif


namespace NNSimulator {

    template<class T> class Data;

    template<class T> class SolverPCNNI2003E;

    template<class T> class SolverImpl;
    template<class T> class SolverImplCPU;
    template<class T> class SolverImplCuda;

    //! Базовый класс для численных решателей.
    template<class T> class Solver
    {
            //! Тип вектора спайков.
            using vSpikes = std::vector<std::pair<size_t,T>>;

            //! Константный итератор для вектора спайков.
            using citSpikes = typename vSpikes::const_iterator;

            //! Тип вектора осцилограмм.
            using vOscillograms = std::vector<std::pair<size_t,std::valarray<T>>>;

            //! Константный итератор для вектора осцилограмм.
            using citOscillograms = typename vOscillograms::const_iterator;

        protected:

            //! Номер решателя.
            const size_t sNum_;

            //! Номер набора данных.
            const size_t dNum_ ;

            //! Указатель на данные. 
            std::unique_ptr<Data<T>> pData_ {nullptr};

            //! Указатель на реализацию. 
            std::unique_ptr<SolverImpl<T>> pImpl_ {nullptr};

            //! Определяет интерфейс метода для вызова решающего метода из установленной реализации.
            virtual void solveImpl_( const T & cst ) = 0;

            //! Вектор для хранения диаграммы спайков
            vSpikes spikes_;

            //! Вектор для хранения осцилограмм
            vOscillograms oscillograms_;

    public:

            //! Конструктор.
            explicit Solver( const size_t & dNum, const size_t & sNum ) :  
                sNum_(sNum),
                dNum_(dNum),
                pData_
                (
                    Data<float>::createItem(
                        static_cast<typename Data<T>::ChildId>( dNum_ )
                    )
                ),
                pImpl_
                (
                    #ifdef NN_CUDA_IMPL
                        std::make_unique<SolverImplCuda<T>>()
                    #else 
                        std::make_unique<SolverImplCPU<T>>()
                    #endif
                )
            {}

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
                SolverPCNNI2003EId = 1, //!< Е.М. Ижикевич, 2003, метод Эйлера 
            };

            //! Фабричный метод создания конкретного решателя. 
            static std::unique_ptr<Solver<T>> createItem( ChildId id )
            {
                std::unique_ptr<Solver<T>> ptr;
                switch( id )
                {
                    case SolverPCNNI2003EId:
                        ptr = std::unique_ptr<Solver<T>>( std::make_unique<SolverPCNNI2003E<T>>() );
                    break;
                }
                return ptr;
            }
            

            void solve( std::ostream && ostr = std::ostream(nullptr) )
            {
                T cte = pData_->t;
                if( ostr ) ostr << *this << std::endl;
                cte += pData_->dtDump;
                while( pData_->t < pData_->tEnd ) 
                {
                    solveImpl_( cte );
                    cte = pData_->t;
                    if( ostr ) ostr << *this << std::endl;
                    cte += pData_->dtDump;
                } 
            }

            //! Потоковое чтение данных.
            virtual std::istream& read( std::istream& istr ) = 0 ;

            //! Потоковая запись данных.
            virtual std::ostream& write( std::ostream& ostr ) const = 0 ;

            //! Возвращает итераторы на начало и конец вектора спайков.
            std::pair<citSpikes,citSpikes> getSpikes() const
            {
                return std::pair<citSpikes,citSpikes>
                    (
                         spikes_.cbegin(),
                         spikes_.cend()
                    );
            }

            //! Возвращает итераторы на начало и конец вектора с осцилограммами.
            std::pair<citOscillograms,citOscillograms> getOscillograms() const
            {
                return std::pair<citOscillograms,citOscillograms>
                    (
                         oscillograms_.cbegin(),
                         oscillograms_.cend()
                    );
            }
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

namespace std{

    template<class T>
    std::ostream& operator<<( std::ostream & ostr, const std::pair<size_t,T> & p)
    {
        return (ostr<<p.first << ' ' << p.second);
    }

    template<class T>
    std::ostream& operator<<( std::ostream & ostr, const std::pair<size_t,std::valarray<T>> & p)
    {
        ostr << p.first << '\t';
        for( const auto & e : p.second )
            ostr << e << ' ';
    }
}

#endif

/*@}*/

