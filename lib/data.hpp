/** @addtogroup Data
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorDataNDN2017_
#define _NeuroglialNetworkSimulatorDataNDN2017_

#include <iostream>
#include <valarray>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "formatstream.hpp"
#include "../tools/binarybuffer.h"
#include "graphinformation.h"

namespace NNSimulator {

    //! \~russian \brief Базовый класс для хранения данных, используемых при симуляции.
    template<class T> struct Data
    {
    protected:
    	/*! \~russian
    	 * \brief Вспомогательная функция для печати элементов контейнера vector или valarray.
    	 * \details Если у вектора есть элементы, печатает их в строку через пробел. В конце добавляет
    	 * два символа переноса строки.
    	 * \param vec константная ссылка на вектор, который должен быть напечатан.
    	 * \param fout ссылка на поток, куда будет напечатана информация.
    	 */
    	template<typename U>
    	static void printVector( const U& vec, std::ostream& fout ){
    		if ( vec.size() > 0 ){
    			fout << vec[0];
    			for(size_t i=1; i < vec.size(); ++i){
    				fout << ' ' << vec[i];
    			}
    			fout << std::endl << std::endl;
    		}
    	}

    	/*! \~russian
    	 * \brief Вспомогательная функция для печати элементов контейнера vector или valarray,
    	 * в которых информация хранится как будто в матрице.
    	 * \details Если у матрицы есть элементы, печатает их в строку через пробел.
    	 * Каждая строка матрицы печатается с новой строки.
    	 * В конце добавляет два символа переноса строки.
    	 * \param vec константная ссылка на вектор, который должен быть напечатан.
    	 * \param Ncolumns количество колонок в матрице.
    	 * \param fout ссылка на поток, куда будет напечатана информация.
    	 */
    	template<typename U>
    	static void printMatrix( const U& vec, size_t Ncolumns, std::ostream& fout ){
    		if ( vec.size() > 0 ){
    			size_t Nrows = vec.size() / Ncolumns;
    			if ( ( vec.size() % Ncolumns ) != 0 ){
    				std::cerr << "ERROR in printMatrix( const U& vec, size_t Ncolumns, std::ostream& fout ) : ";
    				std::cerr << __LINE__ << ", file " << __FILE__ << std::endl;
    				std::cerr << "\tNrows * Ncolumns != vec.size()" << std::endl;
    				std::cerr << "\tNrows = " << Nrows << std::endl;
    				std::cerr << "\tNcolumns = " << Ncolumns << std::endl;
    				std::cerr << "\tvec.size() = " << vec.size() << std::endl;
    				throw;
    			}

    			for(size_t i=0; i < Nrows; ++i){
    				fout << vec[i*Ncolumns];
    				for(size_t k=1; k < Ncolumns; ++k){
    					fout << ' ' << vec[k + i*Ncolumns];
    				}
    				fout << std::endl;
    			}
    			fout << std::endl;
    		}
    	}

    public:
            //! \~russian \brief Флаг вывода в бинарном виде. Если true, то выходной файл будет содержать данные в бинарном формате.
    		bool isBinaryWrite_ {false};
    		//! \~russian \brief Флаг считывания из потока в бинарном виде. Если true, то входной файл должен быть записан в бинарном формате.
    		bool isBinaryRead_ {false};

            //! \~russian \brief Число нейронов.
            size_t nNeurs {0};

            //! \~russian \brief Число возбуждающих нейронов.
            size_t nNeursExc {0};

            //! \~russian \brief Вектор мембранных потенциалов. \details Является переменной. Меняется каждый временной шаг.
            std::valarray<T> VNeurs {};

            //! \~russian \brief Предельное значение потенциала.
            T VNeursPeak {};

            //! \~russian \brief Значение потенциала после спайка.
            T VNeursReset {};

            //! \~russian \brief Маска, хранящая спайки. \details Меняется каждый временной шаг в модели Е.М. Ижикевича.
            std::valarray<bool> mNeurs {};

            //! \~russian \brief Вектор токов для нейров. \details Меняется каждый временной шаг в модели Е.М. Ижикевича.
            std::valarray<T> INeurs {};

            //! \~russian \brief Матрица весов nNeurs_ x nNeurs_.
            std::valarray<T> wConns {};

            //! \~russian \brief Модельное время.
            T t {0};

            //! \~russian \brief Шаг по времени.
            T  dt {0}; 

            //! \~russian \brief Время симуляции.
            T tEnd {0}; 

            //! \~russian \brief Временной период для сохранения дампа.
            T dtDump {0};

            //! \~russian \brief Конструктор по умолчанию.
            explicit Data() = default;

            //! \~russian \brief Деструктор по умолчанию.
            virtual ~Data() = default;

            //! \~russian \brief Копирующий конструктор по умолчанию.
            Data( const Data& otherData) = default;

            //! \~russian \brief Оператор присваивания по умолчанию.
            Data& operator=( const Data& rhsData ) = default;

            //! \~russian \brief Перемещающий конструктор по умолчанию.
            Data( Data&& otherData ) = default;

            //! \~russian \brief Перемещающий оператор присваивания по умолчанию.
            Data& operator=( Data&& rhsData ) = default;

            //! \~russian \brief Перечисление с типами данных для различных моделей.
            enum ChildId : size_t
            {  
                DataPCNNI2003Id = 1, //!< \~russian модель Е.М. Ижикевича 2003
                DataUNN270117Id = 2    //!< \~russian модель ННГУ 27.01.17
            };

            /*! \~russian
             * \brief Фабричный метод создания конкретного набора данных.
             * \param id идентификационный номер типа структуры данных.
             * \return уникальный указатель на структуру данных типа, указанного в аргументе функции.
             */
            static std::unique_ptr<Data<T>> createItem( ChildId id );

            /*! \~russian
             * \brief Функция устанавливает флаг формата выходного буфера (файла).
             * \details Если isbin равен true, то выходной файл будет содержать
             * данные в бинарном формате.
             * \param isbin новый флаг для формата выходного буфера (файла).
             */
            void setIsBinaryWriteFlag(bool isbin) { isBinaryWrite_ = isbin; }

            /*! \~russian
             * \brief Функция устанавливает флаг формата входного буфера (файла).
             * \param isbin новый флаг для формата входного буфера (файла).
             */
            void setIsBinaryReadFlag(bool isbin) { isBinaryRead_ = isbin; }

            /*! \~russian
             * \brief Функция возвращает текущий флаг формата выходных данных.
             * \return флаг формата выходных данных.
             */
            bool isBinaryWrite() const { return isBinaryWrite_; }

            /*! \~russian
             * \brief Функция возвращает текущий флаг формата для входных данных.
             * \return флаг формата для входных данных (из потока).
             */
            bool isBinaryRead() const { return isBinaryRead_; }

            /*! \~russian
             * \brief Функция возвращает структуру с необходимой информацией для создания графиков для всех параметров.
             * \return вектор с параметрами для создания графиков.
             */
            virtual std::vector<typename NNSimulator::GraphInformation> getGraphNames(const std::string& mainPartOfName) const = 0;

            /*! \~russian
             * \brief Функция преобразует данные в текущей структуре в данные для графика и выводит их.
             * \details Поддерживает бинарный вывод.
             * \param streams ссылка на вектор с указателями на потоки.
             * Все элементы вектора должны располагаться в том же порядке, в каком выдает их функция
             * getGraphNames().
             */
            virtual void writeDataInGraphics( std::vector<std::unique_ptr<std::ofstream>>& streams ) const = 0;

            /*! \~russian
             * \brief Потоковое чтение данных.
             * \details Используется для чтения из входного файла симулятора.
             * \param[out] istr ссылка на поток ввода, откуда берутся данные.
             * \return ссылку на поток ввода, откуда были взяты данные.
             */
            virtual std::istream& read( std::istream& istr ) 
            {
            	if ( isBinaryRead_ ){
            		BinaryBufferInS binin( istr );
            		// solver
            		binin >> t >> tEnd >> dt >> dtDump;
            		// neurs
            		binin >> nNeurs >> nNeursExc;

            		VNeurs.resize( nNeurs );
            		mNeurs.resize( nNeurs );

            		binin >> VNeurs >> mNeurs >> VNeursPeak >> VNeursReset;

            		// conns
            		INeurs.resize( nNeurs );
            		wConns.resize( nNeurs * nNeurs );
            		binin >> INeurs >> wConns;
            	}
            	else{
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
            	}
                return istr;
            }

            /*! \~russian
             * \brief Потоковая запись данных.
             * \details Используется для регулярной записи в выходной файл симулятора.
             * \param ostr ссылка на поток вывода, куда будут напечатаны данные структуры.
             * \return ссылку на поток вывода, куда были напечатаны данные структуры.
             */
            virtual std::ostream& write( std::ostream& ostr ) const 
            {
            	if (isBinaryWrite_){
					#if DEBUG >= 2
            			std::cout << "is binary true in Data. writing in binary buff..." << std::endl;
					#endif
            		BinaryBufferOutS binbuf( ostr );
            		binbuf << t << tEnd << dt << dtDump << nNeurs << nNeursExc
            			   << VNeurs << mNeurs << VNeursPeak << VNeursReset
						   << INeurs << wConns;
            	}
            	else{
					#if DEBUG >= 2
            			std::cout << "is binary false in Data. writing in usual buff..." << std::endl;
					#endif
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
            	}
				#if DEBUG >= 2
            		std::cout << "in Data. writing in buff complete" << std::endl;
				#endif
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
            	const std::string sscalar = "#scalar ";
            	const std::string smatrixname = "#matrixname ";
            	const std::string smatrix = "#matrix ";
            	const std::string svector = "#vector ";

            	fout << sscalar      << "simNum" << std::endl;
            	fout << static_cast<int>( Data<T>::ChildId::DataPCNNI2003Id ) << std::endl;
            	fout << sscalar      << "time" << std::endl;
            	fout << t            << std::endl;
            	fout << sscalar      << "timeEnd" << std::endl;
            	fout << tEnd         << std::endl;
            	fout << sscalar      << "deltaTime" << std::endl;
            	fout << dt           << std::endl;
            	fout << sscalar      << "deltaTimeForDump" << std::endl;
            	fout << dtDump       << std::endl;
            	fout << sscalar      << "numberOfNeurons" << std::endl;
            	fout << nNeurs       << std::endl;
            	fout << sscalar      << "numberOfExcitatoryNeurs" << std::endl;
            	fout << nNeursExc    << std::endl << std::endl;

            	fout << svector 	 << "VNeurs" << std::endl;
            	printVector( VNeurs, fout );
            	fout << svector      << "m" << std::endl;
            	printVector( mNeurs, fout );
            	fout << sscalar      << "VNeursPeak" << std::endl;
            	fout << VNeursPeak   << std::endl;
            	fout << sscalar      << "VNeursReset" << std::endl;
            	fout << VNeursReset  << std::endl;
            	fout << svector      << "INeurs" << std::endl;
            	printVector( INeurs, fout );

            	fout << smatrixname  << "weightsOfConns" << std::endl;
            	fout << smatrix;
            	fout << '1';
            	for( size_t i=1; i < nNeurs; ++i ){
            		fout << ' ' << i+1;
            	}
            	fout << std::endl;
            	printMatrix( wConns, nNeurs, fout );

            	return fout;
            }

            /*! \~russian
             * \brief Функция проверяет корректность данных, которые
             * находятся в потоке istr.
             * \details Данные в потоке проверяются на то, подходят ли они
             * для входного файла симулятора.
             * \param istr ссылка на поток, откуда ведется считывание.
             * \return true, если данные подходят для входного файла симулятора.
             */
            virtual std::pair<bool, std::string> checkFile( std::istream& istr ) const = 0;

    };

} // namespace


/*! \~russian
 * \brief Оператор потокового вывода.
 * \details Использует открытый метод write().
 * \param ostr ссылка на поток вывода, куда будет напечатана информация о структуре.
 * \param item элемент, данные которого будут напечатаны.
 * \return ссылку на поток вывода, куда были напечатаны данные структуры.
 */
template<class T>
inline std::ostream& operator<<( std::ostream & ostr, const NNSimulator::Data<T> & item)
{
    return (item.write(ostr));
}

/*! \~russian
 * \brief Оператор потокова ввода.
 * \details Использует открытый метод read().
 * \param istr ссылка на поток ввода, откуда берутся данные для записи в структуру.
 * \param item элемент, параметры которого заполняются в результате считывания.
 * \return ссылку на поток ввода, откуда были считаны данные.
 */
template<class T>
inline std::istream& operator>>( std::istream & istr, NNSimulator::Data<T> & item){
    return (item.read(istr));
}

#endif

/*@}*/

