/** @addtogroup Neurs
 * @{*/

/** @file */

#ifndef _NNetworkSimulatorNeursNDN2017_
#define _NNetworkSimulatorNeursNDN2017_

#include <iostream>
#include <iomanip>
#include <valarray>
#include <memory>

#include "neursspec.hpp"

#include "setting.h"

#ifdef NN_CUDA_IMPL 
    #include "./impl/cuda/neurscuda.hpp"
#else 
    #include "./impl/cpu/neurscpu.hpp"
#endif

namespace NNSimulator {

    template<class T> class NeursImpl;
    template<class T> class NeursImplCPU;
    template<class T> class NeursImplCuda;

    template<class T> class NeursSpec;

    //! Базовый класс для хранения общих данных нейронов.
    template<class T> class Neurs {
 
        protected:

            //! Число нейронов в моделируемой сети.
            size_t N_ {0};

            //! Вектор мембранных потенциалов.
            std::valarray<T> V_;

            //! Маска, хранящая спайки.
            std::valarray<bool> mask_;

            //! Модельное время 
            T t_ ;

            //! Указатель на массив токов.
            const std::valarray<T> *I_ {nullptr};

            //! Разыменование указателя на массив токов. 
            const std::valarray<T> & I() noexcept { return *I_; }

            //! Указатель на реализацию. 
            std::unique_ptr<NeursImpl<T>> pImpl;

        public:

            //! Конструктор.
            explicit Neurs() :  pImpl( 
                #ifdef NN_CUDA_IMPL
                    std::make_unique<NeursImplCuda<T>>() 
                #else 
                    std::make_unique<NeursImplCPU<T>>() 
                #endif
            ) { }

            //! Деструктор.
            virtual ~Neurs() = default;

            //! Копирующий конструктор.
            Neurs( const Neurs& ) = delete;

            //! Оператор присваивания.
            Neurs& operator=( const Neurs& ) = delete;

            //! Перемещающий конструктор.
            Neurs( const Neurs&& ) = delete;

            //! Перемещающий оператор присваивания.
            Neurs& operator=( const Neurs&& ) = delete;

            //! Вычисляет изменение состояния нейронов за промежуток времени \f$dt\f$.
            virtual void performStepTime( const T & dt ) = 0;

            //! Устанавливает указатель на вектор со значениями синаптических токов.
            void setCurrents( std::valarray<T> & I ) { I_ = &I; } // !!!!!

            //! Перечисление с типами моделей нейронов.
            enum ChildId : size_t
            {  
                NeursSpecId = 0, //!< идентификатор для тестового класса.
                NeursIzhikId = 1, //!< идентификатор для нейрона Ижикевича.
                NeursHHId = 2 //!< идентификатор для нейрона Х.-Х.
            };

            //! Фабричный метод создания новых объектов. \details 
            std::unique_ptr<Neurs<T>> createItem( ChildId id )
            {
                std::unique_ptr<Neurs<T>> ptr;
                switch( id )
                {
                    case NeursSpecId:
                        ptr = std::unique_ptr<Neurs<T>>( std::make_unique<NeursSpec<T>>() );
                    break;
                    case NeursIzhikId:
                    //     p = std::shared_ptr<Neurs<T>>(new NeursIzhik<T>);
                    break;
                }
                return ptr;
            }

            //! \~russian Метод вывода параметров в поток. 
            virtual std::ostream& write( std::ostream& ostr ) const 
            {
                ostr << N_ << '\t' << t_ << '\t';
                for( const auto & e: V_ ) ostr << std::setw(10) << e << ' ' ;
                ostr << '\t';
                return ostr;
            } 

            //! \~russian Метод ввода параметров из потока.  
            virtual std::istream& read( std::istream& istr ) 
            {
                istr >> N_  >> t_;
                V_.resize(N_);
                for( auto & e: V_ ) istr >> e;
                return istr;
            }

        };
}

//! Оператор потокового вывода.
template<class T>
std::ostream& operator<<( std::ostream & ostr, const NNSimulator::Neurs<T> & item)
{
    return (item.write(ostr));
}

//! Оператор потокова ввода.
template<class T>
std::istream& operator>>( std::istream & istr, NNSimulator::Neurs<T> & item)
{
    return (item.read(istr));
}

#endif

/*@}*/

