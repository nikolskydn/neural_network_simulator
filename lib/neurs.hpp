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
#include "formatstream.hpp"

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

            //! Предельное значение потенциала.
            T VPeak_;

            //! Значение потенциала после спайка.
            T VReset_;

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

            //! Инициализация перед вызовом PerformStepTime().
            virtual void performStepTimeInit() = 0;

            //! Вычисляет изменение состояния нейронов за промежуток времени \f$dt\f$.
            virtual void performStepTime( const T & dt ) = 0;

            //!  Освобождение ресурсов после PerformStepTime().
            virtual void performStepTimeFinalize() = 0;

            //! Устанавливает указатель на вектор со значениями синаптических токов.
            void setCurrents( const std::valarray<T> & I ) { I_ = &I; } // !!!!!

            //! Ссылка на массив потенциалов. 
            const std::valarray<T> & getPotentials() const noexcept { return V_; }

            //! Ссылка на маску спайков. 
            const std::valarray<bool> & getMasks() const noexcept { return mask_; }

            //! Время текущего состояния системы.
            T getTime() const { return t_ ; }

            //! Перечисление с типами моделей нейронов.
            enum ChildId : size_t
            {  
                NeursSpecId = 0, //!< идентификатор для тестового класса.
                NeursIzhikId = 1, //!< идентификатор для нейрона Ижикевича.
                NeursHHId = 2 //!< идентификатор для нейрона Х.-Х.
            };

            //! Фабричный метод создания новых объектов. \details 
            static std::unique_ptr<Neurs<T>> createItem( ChildId id )
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

            //! Метод вывода параметров в поток. \details Порядок записи: - N_ - t_ - V_ - mask_ - VPeak_ - VReset_.
            virtual std::ostream& write( std::ostream& ostr ) const 
            {
                FormatStream oFStr( ostr );
                oFStr << N_ << t_ ;
                for( const auto & e: V_ ) oFStr << e ;  
                for( const auto & e: mask_ ) oFStr << e ;  
                oFStr << VPeak_ <<  VReset_ ;
                return oFStr;
            } 

            //! Метод ввода параметров из потока. \details Порядок чтения: - N_ - t_ - V_ - mask_ - VPeak_ - VReset_. 
            virtual std::istream& read( std::istream& istr ) 
            {
                istr >> N_  >> t_;
                V_.resize(N_);
                mask_.resize(N_);
                for( auto & e: V_ ) istr >> e;
                for( auto & e: mask_ ) istr >> e;
                istr >> VPeak_ >> VReset_;
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

