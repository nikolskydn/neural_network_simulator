/** @addtogroup Data
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorDataPulseCoupledNeuralNetworkIzhik2003NDN2017_
#define _NeuroglialNetworkSimulatorDataPulseCoupledNeuralNetworkIzhik2003NDN2017_

#include <iostream>
#include <valarray>
#include "data.hpp"

namespace NNSimulator {

    template<class T> class Data;

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
    template<class T> struct DataPCNNI2003: public Data<T>
    {

            using Data<T>::nNeurs;
            using Data<T>::nNeursExc;
            using Data<T>::VNeurs;
            using Data<T>::VNeursPeak;
            using Data<T>::VNeursReset;
            using Data<T>::mNeurs;
            using Data<T>::INeurs;
            using Data<T>::wConns;
            using Data<T>::t;
            using Data<T>::tEnd;
            using Data<T>::dt;
            using Data<T>::dtDump;

            //! Вектор мембранной восстановительной переменной \f$U\f$. \details Обеспечивает обратную связь. Определяет активацию ионного тока \f$K^+\f$ и деактивацию ионов \f$Na^+\f$.
            std::valarray<T> UNeurs {};
            
            //! Вектор параметров \f$a\f$ из основной системы ОДУ. \details Определяет временной ммасштаб восстановительной переменной \f$U\f$.
            std::valarray<T> aNeurs {};

            //! Вектор параметров \f$b\f$ из основной системы ОДУ. \details Определяет чувствительность восстановительной переменной \f$U\f$.
            std::valarray<T> bNeurs {};

            //! Вектор для вычисления значений мембранных потенциалов после спайка.
            std::valarray<T> cNeurs {};

            //! Вектор для вычисления значений восстановительной переменной \f$U\f$ после спайка.
            std::valarray<T> dNeurs {};

            //! Конструктор.
            explicit DataPCNNI2003() = default;

            //! Деструктор.
            virtual ~DataPCNNI2003() = default;

            //! Копирующий конструктор.
            DataPCNNI2003( const DataPCNNI2003& ) = delete;

            //! Оператор присваивания.
            DataPCNNI2003& operator=( const DataPCNNI2003& ) = delete;

            //! Перемещающий конструктор.
            DataPCNNI2003( DataPCNNI2003&& ) = delete;

            //! Перемещающий оператор присваивания.
            DataPCNNI2003& operator=( DataPCNNI2003&& ) = delete;


            //! Метод вывода параметров в поток. 
            virtual std::ostream& write( std::ostream& ostr ) const final
            {
                FormatStream oFStr( Data<T>::write( ostr ) );
                //oFStr << sNum_;
                for( const auto & e: UNeurs ) oFStr << e ;  
                for( const auto & e: aNeurs ) oFStr << e ;  
                for( const auto & e: bNeurs ) oFStr << e ;  
                for( const auto & e: cNeurs ) oFStr << e ;  
                for( const auto & e: dNeurs ) oFStr << e ;  
                return ostr;
            }

            //! Метод ввода параметров из потока.  
            virtual std::istream& read( std::istream& istr ) final
            {  
                Data<T>::read(istr);

                UNeurs.resize(nNeurs);
                for( auto & e: UNeurs ) istr >> e;
                aNeurs.resize(nNeurs);
                for( auto & e: aNeurs ) istr >> e;
                bNeurs.resize(nNeurs);
                for( auto & e: bNeurs ) istr >> e;
                cNeurs.resize(nNeurs);
                for( auto & e: cNeurs ) istr >> e;
                dNeurs.resize(nNeurs);
                for( auto & e: dNeurs ) istr >> e;
                return istr;
            }
        };
} // namespace

#endif

/*@}*/
