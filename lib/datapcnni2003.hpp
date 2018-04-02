/** @addtogroup Data
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorDataPulseCoupledNeuralNetworkIzhik2003NDN2017_
#define _NeuroglialNetworkSimulatorDataPulseCoupledNeuralNetworkIzhik2003NDN2017_

#include <iostream>
#include <valarray>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include "../tools/binarybuffer.h"
#include "data.hpp"
#include "graphinformation.h"

namespace NNSimulator {

    template<class T> class Data;
    struct GraphInformation;

    /*! \~russian \brief Класc содержит реализацию модели Е.М. Ижикевича,
     * принадлежащую классу импульсно-связанных нейронных сетей (pulse-coupled neural networks).
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
            using Data<T>::isBinaryRead_;
            using Data<T>::isBinaryWrite_;

            /*! \~russian
             * \brief Вектор мембранной восстановительной переменной \f$U\f$.
             * \details Обеспечивает обратную связь.
             * Определяет активацию ионного тока \f$K^+\f$ и деактивацию ионов \f$Na^+\f$.
             */
            std::valarray<T> UNeurs {};
            
            /*! \~russian
             * \brief Вектор параметров \f$a\f$ из основной системы ОДУ.
             * \details Определяет временной ммасштаб восстановительной переменной \f$U\f$.
             */
            std::valarray<T> aNeurs {};

            /*! \~russian
             * \brief Вектор параметров \f$b\f$ из основной системы ОДУ.
             * \details Определяет чувствительность восстановительной переменной \f$U\f$.
             */
            std::valarray<T> bNeurs {};

            //! \~russian \brief Вектор для вычисления значений мембранных потенциалов после спайка.
            std::valarray<T> cNeurs {};

            //! \~russian \brief Вектор для вычисления значений восстановительной переменной \f$U\f$ после спайка.
            std::valarray<T> dNeurs {};

            //! \~russian \brief Конструктор по умолчанию.
            explicit DataPCNNI2003() = default;

            //! \~russian \brief Деструктор по умолчанию.
            virtual ~DataPCNNI2003() = default;

            //! \~russian \brief Копирующий конструктор по умолчанию.
            DataPCNNI2003( const DataPCNNI2003& rhsData ) = default;

            //! \~russian \brief Оператор присваивания по умолчанию.
            DataPCNNI2003& operator=( const DataPCNNI2003& otherData ) = default;

            //! \~russian \brief Перемещающий конструктор по умолчанию.
            DataPCNNI2003( DataPCNNI2003&& otherData ) = default;

            //! \~russian \brief Перемещающий оператор присваивания по умолчанию.
            DataPCNNI2003& operator=( DataPCNNI2003&& rhsData ) = default;

            /*! \~russian
             * \brief Функция возвращает структуру с необходимой информацией для
             * создания графиков для всех параметров.
             * \details В зависимости от флага бинарного вывода добавляет расширение
             * .animate и .banimate .
             *
             * Также добавляет нижнее подчеркивание вместе с названием параметра.
             * Для модели PCNNI2003 выводит параметр V в виде графика и в виде растра.
             * \param mainPartOfName основное имя, к которому прибавляются расширения.
             * \return вектор с параметрами для создания графиков.
             */
            virtual std::vector<typename NNSimulator::GraphInformation> getGraphNames(const std::string& mainPartOfName) const override{
            	std::vector<typename NNSimulator::GraphInformation> res(2);
            	const std::string xDesc = "t, ms";
            	std::string osks, spks;

            	if ( isBinaryWrite_ ){
            		osks = ".boscs";
            		spks = ".bspks";
            	}
            	else{
            		osks = ".oscs";
            		spks = ".spks";
            	}

            	res[0].filename = mainPartOfName + "_V" + spks;
            	res[0].title = "Neurons\' spikes";
            	res[0].xDesc = xDesc;
            	res[0].yDesc = "N";
            	res[0].numberOfGraphics = 1;

            	res[1].filename = mainPartOfName + "_V" + osks;
            	res[1].title = "Voltage on neurons";
            	res[1].xDesc = xDesc;
            	res[1].yDesc = "V";
            	res[1].numberOfGraphics = nNeurs;
            	return res;
            }

            /*! \~russian
             * \brief Функция преобразует данные в текущей структуре в данные для графика и выводит их.
             * \details Поддерживает бинарный вывод.
             * \param streams ссылка на вектор с указателями на потоки.
             * Все элементы вектора должны располагаться в том же порядке, в каком выдает их функция
             * getGraphNames().
             */
            virtual void writeDataInGraphics( std::vector<std::unique_ptr<std::ofstream>>& streams ) const override{
            	int Nspks = 0;
            	for(size_t i=0; i < nNeurs; ++i){
            		if ( VNeurs[i] >= VNeursPeak )
            			++Nspks;
            	}

            	if ( isBinaryWrite_ ){
            		BinaryBufferOutS binSpks( *streams[0] ), binOsks( *streams[1] );
            		binSpks << t << Nspks;
            		binOsks << t << VNeurs;
            	}
            	else{
            		*streams[0] << t << ' ' << Nspks << std::endl;
            		writeGraphColumnsInText(*streams[1], t, VNeurs);
            	}
            }

            /*! \~russian
             * \brief Метод вывода параметров в поток.
             * \param ostr ссылка на выходной поток, куда будет напечатана информация.
             * \return ссылку на выходной поток, куда была напечатана информация.
             */
            virtual std::ostream& write( std::ostream& ostr ) const final
            {
            	Data<T>::write( ostr );
            	if ( isBinaryWrite_ ){
            		BinaryBufferOutS binout( ostr );

            		binout << UNeurs
            			   << aNeurs
						   << bNeurs
						   << cNeurs
						   << dNeurs;
            	}
            	else{
                    FormatStream oFStr( ostr );
                    //oFStr << sNum_;
                    for( const auto & e: UNeurs ) oFStr << e ;
                    for( const auto & e: aNeurs ) oFStr << e ;
                    for( const auto & e: bNeurs ) oFStr << e ;
                    for( const auto & e: cNeurs ) oFStr << e ;
                    for( const auto & e: dNeurs ) oFStr << e ;
            	}
                return ostr;
            }

            /*! \~russian
             * \brief Функция записывает в поток данные в читаемом для человека формате.
             * \details Использует специальные метки из текстового редактора клиентской
             * части, чтобы файл можно было просматривать с помощью таблиц и вкладок.
             * \param fout ссылка на поток вывода, куда будут напечатаны данные структуры.
             * \return ссылка на поток вывода, куда были напечатаны данные.
             */
            virtual std::ostream& writeAsSin( std::ostream& fout ) const{
            	Data<T>::writeAsSin( fout );

            	const std::string svector = "#vector ";

            	fout << svector << "UNeurs" << std::endl;
            	Data<T>::printVector( UNeurs, fout );
            	fout << svector << "aNeurs" << std::endl;
            	Data<T>::printVector( aNeurs, fout );
            	fout << svector << "bNeurs" << std::endl;
            	Data<T>::printVector( bNeurs, fout );
            	fout << svector << "cNeurs" << std::endl;
            	Data<T>::printVector( cNeurs, fout );
            	fout << svector << "dNeurs" << std::endl;
            	Data<T>::printVector( dNeurs, fout );

            	return fout;
            }

            /*! \~russian
             * \brief Метод ввода параметров из потока.
             * \param istr ссылка на входной поток, откуда берутся данные для записи в структуру.
             * \return ссылку на входной поток, откуда были взяты данные для записи в структуру.
             */
            virtual std::istream& read( std::istream& istr ) final
            {  
                Data<T>::read(istr);

                UNeurs.resize(nNeurs);
                aNeurs.resize(nNeurs);
                bNeurs.resize(nNeurs);
                cNeurs.resize(nNeurs);
                dNeurs.resize(nNeurs);

                if (isBinaryRead_){
                	BinaryBufferInS binin( istr );
                	binin >> UNeurs
						  >> aNeurs
						  >> bNeurs
						  >> cNeurs
						  >> dNeurs;
                }
                else{
                    for( auto & e: UNeurs ) istr >> e;
                    for( auto & e: aNeurs ) istr >> e;
                    for( auto & e: bNeurs ) istr >> e;
                    for( auto & e: cNeurs ) istr >> e;
                    for( auto & e: dNeurs ) istr >> e;
                }
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
            	using mess = std::pair<bool, std::string>;
            	T vfloat;
            	bool vbool;

            	auto toStr = [](const T& elem) -> std::string{
            		std::stringstream vss;
            		vss << elem;
            		return vss.str();
            	};

				#ifndef READPARAMMACROSES
				#define READPARAMMACROSES
					#define READPARAM(p, s) if ( !(istr >> p) ) \
						return mess(false, std::string("Small amount of parameters. Can\'t read ") + s);
					#define READPARAMBIN(p, s) if ( !(binin >> p) ) \
						return mess(false, std::string("Small amount of parameters. Can\'t read ") + s);
				#endif

            	if ( isBinaryRead_ ){
            		T vt, vtEnd, vdt, vdtDump;

            		BinaryBufferInS binin( istr );
            		// solver
            		READPARAMBIN(vt, "t")
            		READPARAMBIN(vtEnd, "tEnd")
					READPARAMBIN(vdt, "dt")
					READPARAMBIN(vdtDump, "dtDump")

            		if ( vt > vtEnd ){
            			return mess(false, "parameter t must be less than tEnd. Now t = " + toStr(vt) + ", tEnd = " + toStr(vtEnd));
            		}
            		if ( vdt > vdtDump ){
            			return mess(false, "parameter dt must be less than dtDump. Now dt = " + toStr(vdt) + ", dtDump = " + toStr(vdtDump));
            		}
            		if ( vdt <= 0.0 ){
            			return mess(false, "parameter dt must be positive. Now dt = " + toStr(vdt));
            		}
            		if ( vdtDump <= 0.0 ){
            			return mess(false, "parameter dtDump must be positive. Now dtDump = " + toStr(vdtDump));
            		}

            		// neurs
            		long long int vnNeurs, vnNeursExc;
            		READPARAMBIN(vnNeurs, "nNeurs")
            		READPARAMBIN(vnNeursExc, "nNeursExc")
            		if ( vnNeurs <= 0 ){
            			return mess(false, "parameter nNeurs (number of neurons) must be more than 0. Now nNeurs = " + toStr(vnNeurs));
            		}
            		if ( vnNeursExc <= 0 ){
            			return mess(false, "parameter nNeursExc (number of excited neurons) must be more than 0. Now nNeursExc = " + toStr(vnNeursExc));
            		}
            		if ( vnNeursExc > vnNeurs ){
            			return mess(false, "parameter nNeursExc (number of excited neurons) must be less than nNeurs (number of neurons)."
            					"Now nNeursExc = " + toStr(vnNeursExc) + ", nNeurs = " + toStr(vnNeurs));
            		}

                	for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vfloat, "VNeurs["+toStr(i)+"]" )	// VNeurs
					for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vbool, "mNeurs["+toStr(i)+"]" )	// mNeurs
					READPARAMBIN( vfloat, "VNeursPeak" )		// VNeursPeak
					READPARAMBIN( vfloat, "VNeursReset" )		// VNeursReset
					for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vfloat, "INeurs["+toStr(i)+"]" )	// INeurs
					for( long long int i=0; i < vnNeurs*vnNeurs; ++i ) READPARAMBIN( vfloat, "wConns["+toStr(i)+"]" )	// wConns

					// UNeurs, aNeurs, bNeurs, cNeurs, dNeurs
					for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vfloat, "UNeurs" )
					for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vfloat, "aNeurs" )
					for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vfloat, "bNeurs" )
					for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vfloat, "cNeurs" )
					for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vfloat, "dNeurs" )
            	}
            	else{
            		T vt, vtEnd, vdt, vdtDump;

            		// solver
            		READPARAM(vt, "t")
            		READPARAM(vtEnd, "tEnd")
					READPARAM(vdt, "dt")
					READPARAM(vdtDump, "dtDump")

            		if ( vt > vtEnd ){
            			return mess(false, "parameter t must be less than tEnd. Now t = " + toStr(vt) + ", tEnd = " + toStr(vtEnd));
            		}
            		if ( vdt > vdtDump ){
            			return mess(false, "parameter dt must be less than dtDump. Now dt = " + toStr(vdt) + ", dtDump = " + toStr(vdtDump));
            		}
            		if ( vdt <= 0.0 ){
            			return mess(false, "parameter dt must be positive. Now dt = " + toStr(vdt));
            		}
            		if ( vdtDump <= 0.0 ){
            			return mess(false, "parameter dtDump must be positive. Now dtDump = " + toStr(vdtDump));
            		}

            		// neurs
            		long long int vnNeurs, vnNeursExc;
            		READPARAM(vnNeurs, "nNeurs")
            		READPARAM(vnNeursExc, "nNeursExc")
            		if ( vnNeurs <= 0 ){
            			return mess(false, "parameter nNeurs (number of neurons) must be more than 0. Now nNeurs = " + toStr(vnNeurs));
            		}
            		if ( vnNeursExc <= 0 ){
            			return mess(false, "parameter nNeursExc (number of excited neurons) must be more than 0. Now nNeursExc = " + toStr(vnNeursExc));
            		}
            		if ( vnNeursExc > vnNeurs ){
            			return mess(false, "parameter nNeursExc (number of excited neurons) must be less than nNeurs (number of neurons)."
            					"Now nNeursExc = " + toStr(vnNeursExc) + ", nNeurs = " + toStr(vnNeurs));
            		}

                	for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vfloat, "VNeurs["+toStr(i)+"]" )	// VNeurs
					for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vbool, "mNeurs["+toStr(i)+"]" )	// mNeurs
					READPARAM( vfloat, "VNeursPeak" )		// VNeursPeak
					READPARAM( vfloat, "VNeursReset" )		// VNeursReset
					for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vfloat, "INeurs["+toStr(i)+"]" )	// INeurs
					for( long long int i=0; i < vnNeurs*vnNeurs; ++i ) READPARAM( vfloat, "wConns["+toStr(i)+"]" )	// wConns

					// UNeurs, aNeurs, bNeurs, cNeurs, dNeurs
					for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vfloat, "UNeurs" )
					for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vfloat, "aNeurs" )
					for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vfloat, "bNeurs" )
					for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vfloat, "cNeurs" )
					for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vfloat, "dNeurs" )
            	}


				#undef READPARAM
				#undef READPARAMBIN
				#undef READPARAMMACROSES
            	return mess(true, "");
            }
        };
} // namespace

#endif

/*@}*/
