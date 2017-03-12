/** @addtogroup Conns
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorConnsSpecNDN2017_
#define _NeuroglialNetworkSimulatorConnsSpecNDN2017_

#include "conns.hpp"

namespace NNSimulator {

    template<class T> class Conns;

    //! Нейрон со специальным параметром для тестирования. \details Класс предназначен для демонстрации создания различных моделей синаптических связей.
    template<class T> class ConnsSpec : public Conns<T> {

        using Conns<T>::t_;
        using Conns<T>::N_;
        using Conns<T>::I_;
        using Conns<T>::V;
        using Conns<T>::mask;
        //using Conns<T>::V_;
        //using Conns<T>::mask_;
        using Conns<T>::pImpl;

        //! Некоторый специальный параметр.
        T paramSpec {0.1};

        public:

            //! Конструктор.
            explicit ConnsSpec() :  Conns<T>()  {}

            //! Деструктор.
            ~ConnsSpec() = default;

            //! Копирующий конструктор.
            ConnsSpec( const ConnsSpec& ) = delete;

            //! Оператор присваивания.
            ConnsSpec& operator=( const ConnsSpec& ) = delete;

            //! Перемещающий конструктор.
            ConnsSpec( const ConnsSpec&& ) = delete;

            //! Перемещающий оператор присваивания.
            ConnsSpec& operator=( const ConnsSpec&& ) = delete;

            //! Инициализация перед вызовами performStepTime().
            virtual void performStepTimeInit() final
            {
                pImpl->performStepTimeSpecInit( I_.size() );
            }
             
            //! Вычисляет изменение состояния нейронов за промежуток времени dt.
            virtual void performStepTime(const T & dt) final
            {
                pImpl->performStepTimeSpec( dt, paramSpec, V(), mask(), t_, I_ );
            }

            //!  Освобождение ресурсов после PerformStepTime().
            virtual void performStepTimeFinalize() final 
            {
                pImpl->performStepTimeSpecFinalize();
            }

            //! \~russian Метод вывода параметров в поток. 
            virtual std::ostream& write( std::ostream& ostr ) const final
            {
                return (Conns<T>::write(ostr)  << paramSpec << ' ');
            }

            //! \~russian Метод ввода параметров из потока.  
            virtual std::istream& read( std::istream& istr ) final
            {
                return (Conns<T>::read(istr) >> paramSpec);
            }

    };
}

#endif

/*@}*/
