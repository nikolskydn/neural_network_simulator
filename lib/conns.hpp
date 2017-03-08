/** @addtogroup Conns
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorConnsNDN2017_
#define _NeuroglialNetworkSimulatorConnsNDN2017_

#include <iostream>
#include <iomanip>
#include <valarray>
#include <memory>

#include "connsspec.hpp"

#include "setting.h"

#ifdef NN_CUDA_IMPL 
    #include "./impl/cuda/connscuda.hpp"
#else 
    #include "./impl/cpu/connscpu.hpp"
#endif


namespace NNSimulator {

    template<class T> class ConnsImpl;
    template<class T> class ConnsImplCPU;
    template<class T> class ConnsImplCuda;

    template<class T> class ConnsSpec;

    //! Базовый класс для хранения общих данных синаптических связей (синапсов).
    template<class T> class Conns {
 
        protected:

            //! Число синапсов в моделируемой сети.
            size_t N_ {0};

            //! Вектор токов.
            std::valarray<T> I_;

            //! Модельное время 
            T t_ ;

            //! Указатель на массив потенциалов.
            const std::valarray<T> *V_ {nullptr};

            //! Разыменование указателя на массив потенциалов. 
            const std::valarray<T> & V() noexcept { return *V_; }

            //! Указатель на реализацию. 
            std::unique_ptr<ConnsImpl<T>> pImpl;

        public:

            //! Конструктор.
            explicit Conns() :  pImpl( 
                #ifdef NN_CUDA_IMPL
                    std::make_unique<ConnsImplCuda<T>>() 
                #else 
                    std::make_unique<ConnsImplCPU<T>>() 
                #endif
            ) { }

            //! Деструктор.
            virtual ~Conns() = default;

            //! Копирующий конструктор.
            Conns( const Conns& ) = delete;

            //! Оператор присваивания.
            Conns& operator=( const Conns& ) = delete;

            //! Перемещающий конструктор.
            Conns( const Conns&& ) = delete;

            //! Перемещающий оператор присваивания.
            Conns& operator=( const Conns&& ) = delete;

            //! Вычисляет изменение состояния нейронов за промежуток времени \f$dt\f$.
            virtual void performStepTime( const T & dt ) = 0;

            //! Устанавливает указатель на вектор со значениями потенциалов.
            void setPotentials( const std::valarray<T> & V ) { V_ = &V; } 

            //! Ссылка на массив токов. 
            const std::valarray<T> & getCurrents() const noexcept { return I_; }

            //! Перечисление с типами моделей нейронов.
            enum ChildId : size_t
            {  
                ConnsSpecId = 0, //!< идентификатор для тестового класса.
                ConnsIzhikId = 1, //!< идентификатор для синапсов из модели Ижикевича.
            };

            //! Фабричный метод создания новых объектов. \details 
            std::unique_ptr<Conns<T>> createItem( ChildId id )
            {
                std::unique_ptr<Conns<T>> ptr;
                switch( id )
                {
                    case ConnsSpecId:
                        ptr = std::unique_ptr<Conns<T>>( std::make_unique<ConnsSpec<T>>() );
                    break;
                    case ConnsIzhikId:
                    //     p = std::shared_ptr<Conns<T>>(new ConnsIzhik<T>);
                    break;
                }
                return ptr;
            }

            //! \~russian Метод вывода параметров в поток. 
            virtual std::ostream& write( std::ostream& ostr ) const 
            {
                ostr << N_ << '\t' << t_ << '\t';
                for( const auto & e: I_ ) ostr << std::setw(10) << e << ' ' ;
                ostr << '\t';
                return ostr;
            } 

            //! \~russian Метод ввода параметров из потока.  
            virtual std::istream& read( std::istream& istr ) 
            {
                istr >> N_  >> t_;
                I_.resize(N_);
                for( auto & e: I_ ) istr >> e;
                return istr;
            }

        };

}


//! Оператор потокового вывода.
template<class T>
std::ostream& operator<<( std::ostream & ostr, const NNSimulator::Conns<T> & item)
{
    return (item.write(ostr));
}

//! Оператор потокова ввода.
template<class T>
std::istream& operator>>( std::istream & istr, NNSimulator::Conns<T> & item)
{
    return (item.read(istr));
}


#endif

/*@}*/
