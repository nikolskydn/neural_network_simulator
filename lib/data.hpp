/** @addtogroup Data
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorDataNDN2017_
#define _NeuroglialNetworkSimulatorDataNDN2017_

#include <iostream>
#include <valarray>
#include <memory>
#include "formatstream.hpp"
#include "datapcnni2003.hpp"


namespace NNSimulator {

    //! Базовый класс для хранения данных, используемых при симуляции.
    template<class T> struct Data
    {

            //! Число нейронов. 
            size_t nNeurs {0};

            //! Число возбуждающих нейронов. 
            size_t nNeursExc {0};

            //! Вектор мембранных потенциалов.
            std::valarray<T> VNeurs {};

            //! Предельное значение потенциала.
            T VNeursPeak {};

            //! Значение потенциала после спайка.
            T VNeursReset {};

            //! Маска, хранящая спайки.
            std::valarray<bool> mNeurs {};

            //! Вектор токов для нейров. 
            std::valarray<T> INeurs {};

            //! Матрица весов nNeurs_ x nNeurs_.
            std::valarray<T> wConns {};

            //! Модельное время. 
            T t {0};

            //! Шаг по времени.
            T  dt {0}; 

            //! Время симуляции.
            T tEnd {0}; 

            //! Временной период для сохранения дампа.
            T dtDump {0};

            //! Конструктор.
            explicit Data() = default;

            //! Деструктор.
            virtual ~Data() = default;

            //! Копирующий конструктор.
            Data( const Data& ) = delete;

            //! Оператор присваивания.
            Data& operator=( const Data& ) = delete;

            //! Перемещающий конструктор.
            Data( const Data&& ) = delete;

            //! Перемещающий оператор присваивания.
            Data& operator=( const Data&& ) = delete;

            //! Перечисление с типами данных.
            enum ChildId : size_t
            {  
                DataPCNNI2003Id = 1, //!< модель Е.М. Ижикевича 2003
            };

            //! Фабричный метод создания конкретного набора данных. 
            static std::unique_ptr<Data<T>> createItem( ChildId id )
            {
                std::unique_ptr<Data<T>> ptr;
                switch( id )
                {
                    case DataPCNNI2003Id:
                        ptr = std::unique_ptr<Data<T>>( std::make_unique<DataPCNNI2003<T>>() );
                    break;
                }
                return ptr;
            }

            //! Потоковое чтение данных.
            virtual std::istream& read( std::istream& istr ) 
            {
                // solver
                //istr >> sNum;
                istr >> t;
                istr >> tEnd;
                istr >> dt;
                istr >> dtDump;
                // neurs
                istr >> nNeurs ;
                istr >> nNeursExc ;
                VNeurs.resize(nNeurs);
                mNeurs.resize(nNeurs);
                for( auto & e: VNeurs ) istr >> e;
                for( auto & e: mNeurs ) istr >> e;
                istr >> VNeursPeak >> VNeursReset;
                // conns
                INeurs.resize(nNeurs);
                for( auto & e: INeurs ) istr >> e;
                size_t nConns = nNeurs * nNeurs;
                wConns.resize(nConns);
                for( auto & e: wConns ) istr >> e;
                return istr;
            }

            //! Потоковая запись данных.
            virtual std::ostream& write( std::ostream& ostr ) const 
            { 
                FormatStream oFStr( ostr );
                // solver
                oFStr << t;
                oFStr << tEnd;
                oFStr << dt;
                oFStr << dtDump;
                // neurs
                oFStr << nNeurs;
                oFStr << nNeursExc;
                for( const auto & e: VNeurs ) oFStr << e ;  
                for( const auto & e: mNeurs ) oFStr << e ;  
                oFStr << VNeursPeak <<  VNeursReset ;
                // conns
                for( const auto & e: INeurs ) oFStr << e ;
                for( const auto & e: wConns ) oFStr << e  ;
                return oFStr;
            }

    };

} // namespace


//! Оператор потокового вывода.
template<class T>
std::ostream& operator<<( std::ostream & ostr, const NNSimulator::Data<T> & item)
{
    return (item.write(ostr));
}

//! Оператор потокова ввода.
template<class T>
std::istream& operator>>( std::istream & istr, NNSimulator::Data<T> & item)
{
    return (item.read(istr));
}

#endif

/*@}*/

