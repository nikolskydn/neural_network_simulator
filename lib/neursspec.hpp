/** @addtogroup Neurs
 * @{*/

/** @file */

#ifndef _NNetworkSimulatorNeursSpecNDN2017_
#define _NNetworkSimulatorNeursSpecNDN2017_

#include "neurs.hpp"

#include <iostream>

namespace NNSimulator {

    template<class T> class Neurs;

    //! Нейрон со специальным параметром для тестирования. \details Класс предназначен для демонстрации создания различных моделей нейрона.
    template<class T> class NeursSpec : public Neurs<T> {

        using Neurs<T>::t_;
        using Neurs<T>::N_;
        using Neurs<T>::V_;
        using Neurs<T>::VPeak_;
        using Neurs<T>::VReset_;
        using Neurs<T>::mask_;
        using Neurs<T>::I;
        using Neurs<T>::I_;
        using Neurs<T>::pImpl;

        protected: 

            //! Некоторый специальный параметр.
            T paramSpec_ {0.1};

        public:

            //! Конструктор.
            explicit NeursSpec() :  Neurs<T>()  {}

            //! Деструктор.
            virtual ~NeursSpec() = default;

            //! Копирующий конструктор.
            NeursSpec( const NeursSpec& ) = delete;

            //! Оператор присваивания.
            NeursSpec& operator=( const NeursSpec& ) = delete;

            //! Перемещающий конструктор.
            NeursSpec( const NeursSpec&& ) = delete;

            //! Перемещающий оператор присваивания.
            NeursSpec& operator=( const NeursSpec&& ) = delete;

            //! Инициализация перед вызовами performStepTime().
            virtual void performStepTimeInit() final
            {
                pImpl->performStepTimeSpecInit( V_.size() );
            }
             
            //! Вычисляет изменение состояния нейронов за промежуток времени \f$ dt \f$.
            virtual void performStepTime(const T & dt) final
            {
                pImpl->performStepTimeSpec( dt, I(), VPeak_, VReset_, paramSpec_, t_, V_, mask_ );
            }

            //!  Освобождение ресурсов после PerformStepTime().
            virtual void performStepTimeFinalize() final 
            {
                pImpl->performStepTimeSpecFinalize();
            }

            //! Метод вывода параметров в поток. 
            virtual std::ostream& write( std::ostream& ostr ) const final
            {
                return (Neurs<T>::write(ostr)  << paramSpec_ << ' ');
            }

            //! Метод ввода параметров из потока.  
            virtual std::istream& read( std::istream& istr ) final
            {
                return (Neurs<T>::read(istr) >> paramSpec_);
            }

    };
}

#endif

/*@}*/

