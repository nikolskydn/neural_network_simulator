/** @addtogroup Solvers
 * @{*/

/** @file */

#ifndef _NeuroglialNetworkSimulatorSolverNDN2017_
#define _NeuroglialNetworkSimulatorSolverNDN2017_

#include <iostream>
#include <sstream>
#include <valarray>
#include <deque>
#include <iomanip>
#include <type_traits>
#include "formatstream.hpp"

#include "data.hpp"
#include "datacreator.hpp"
#include "solverpcnni2003e.hpp"
#include "solverunn270117.hpp"

#include "setting.h"
#ifdef NN_CUDA_IMPL 
    #include "./impl/cuda/solvercuda.hpp"
#else 
    #include "./impl/cpu/solvercpu.hpp"
#endif


namespace NNSimulator {

    template<class T> class Data;

    template<class T> class SolverPCNNI2003E;
    template<class T> class SolverUNN270117;
    template<class T> class SolverUNN270117e;

    template<class T> class SolverImpl;
    template<class T> class SolverImplCPU;
    template<class T> class SolverImplCuda;

    /*! \~russian \brief Базовый класс для численных решателей.
     *
     * Класс представляет собой интерфейс для решателей.
     * Класс-наследник должен определить функции для считывания и записи
     * данных (read() и write()), функцию вызова имплементации (solveImpl_()).
     *
     * Данный класс имеет фабричный метод для создания необходимого решателя
     * по его номеру.
     *
     * Также есть функция solve(), которая позволяет
     * провести все вычисления на заданном в модели промежутке времени, а также
     * отправляет регулярно информацию о прогрессе в таблицу в базе данных.
     *
     * Схема использования:
     * \code
     *
     * // connect to Database
     * std::unique_ptr<SQLSimulatorReader> dbPtr = nullptr;
     * dbPtr = std::make_unique<SQLSimulatorReader>();
     * dbPtr -> useDefaultParams();
     * dbPtr -> connectToDataBase();
     * dbPtr -> setID( argv[1] );
     *
     * // names of in and out files
     * std::string inFileName = "/common/atlas3d/data/atlastest/testFile.sin";
     * std::string outFileName = "/common/atlas3d/data/atlastest/testFile.sout";
     * std::string spksFileName = "/common/atlas3d/data/atlastest/testFile.spks";
     * std::string oscsFileName = "/common/atlas3d/data/atlastest/testFile.osc";
     * std::string casFileName = "/common/atlas3d/data/atlastest/testFile.Ca";
     *
     * // reading information from .sin file
     * SolversBuffer inBuff;
     * inBuff.readFile(inFileName);

     * size_t outNumberSolver;
     * inBuff >> outNumberSolver;
     *
     * auto s = NNSimulator::Solver<float>::createItem(
     *      static_cast<typename NNSimulator::Solver<float>::ChildId>( outNumberSolver )
     * );
     * s->read( inBuff );
     *
     * // solving equations
     * std::ofstream outFileStream( outFileName );
     * s->solve( std::move(outFileStream), std::move(dbPtr) );
     *
     * // writing necessary information
     * auto citsSpikes = s->getSpikes();
     * std::ofstream spksFileStream(spksFileName);
     * std::copy
     * (
     *   citsSpikes.first,
     *   citsSpikes.second,
     *   std::ostream_iterator<std::pair<size_t,float>>( spksFileStream, "\n" )
     * );
     *
     * auto citsOscillograms = s->getOscillograms();
     * std::ofstream oscsFileStream( oscsFileName );
     * std::copy
     * (
     *   citsOscillograms.first,
     *   citsOscillograms.second,
     *   std::ostream_iterator<std::pair<float,std::valarray<float>>>( oscsFileStream, "\n" )
     * );
     *
     * auto caOsc = s->getCa();
     * std::ofstream caFileStream( casFileName );
     * std::copy
     * (
     *   caOsc.first,
     *   caOsc.second,
     *   std::ostream_iterator<std::pair<float,std::valarray<float>>>( caFileStream, "\n" )
     * );
     *
     * \endcode
     *
     */
    template<class T> class Solver
    {
    	public:
            //! \~russian \brief Тип вектора спайков. \details first - номер нейрона, который спайкнулся; second - время спайка.
            using vSpikes = std::deque<std::pair<T,size_t>>;

            //! \~russian \brief Тип константного итератора для вектора спайков.
            using citSpikes = typename vSpikes::const_iterator;

            /*! \~russian
             * \brief Тип вектора осцилограмм
             * \details В паре first - время записи вектора;
             * second - весь вектор значений параметра на нейронах.
             *
             * Запись в дек происходит каждый шаг по времени.
             */
            using vOscillograms = std::deque<std::pair<T,std::valarray<T>>>;

            //! \~russian \brief Тип константного итератора для вектора осцилограмм.
            using citOscillograms = typename vOscillograms::const_iterator;

        protected:
            //! \~russian \brief Флаг вывода в бинарном виде. Если true, то выходной файл должен быть выведен в бинарном формате.
    		bool isBinaryWrite_ {false};
    		//! \~russian \brief Флаг ввода информации в солвер из бинарного формата. Если true, то входной файл должен быть представлен в бинарном формате.
    		bool isBinaryRead_ {false};
            //! \~russian \brief Номер решателя. \details 1 - решатель модели PCNNI2003, 2 - решатель модели UNN270117.
            const size_t sNum_;

            //! \~russian \brief Номер набора данных. \details 1 - данные модели PCNNI2003, 2 - данные модели UNN270117.
            const size_t dNum_ ;

            //! \~russian \brief Указатель на данные. \details Данные содержат все необходимые параметры и переменные, связанные с моделью.
            std::unique_ptr<Data<T>> pData_ {nullptr};

            //! \~russian \brief Указатель на реализацию. \details Содержит функцию выполнения одного временного шага.
            std::unique_ptr<SolverImpl<T>> pImpl_ {nullptr};

            /*! \~russian
             * \brief Определяет интерфейс метода для вызова решающего метода из установленной реализации.
             * \param cst текущее модельное время.
             */
            virtual void solveImpl_( const T & cst ) = 0;

            //! \~russian \brief Вектор для хранения диаграммы спайков.
            vSpikes spikes_;

            //! \~russian \brief Вектор для хранения осцилограмм.
            vOscillograms oscillograms_;

            //! \~russian \brief Прогресс выполнения.
            size_t progress_ {0};

            //! \~russian \brief Метод создания диаграммы спайков из вектора осцилограмм.
            void makeSpikes_()
            {
                for( const auto & p: oscillograms_ )
                {
                    T t = p.first;
                    for( size_t i=0; i < pData_->nNeurs; ++i )
                        if( p.second[i] >= pData_->VNeursPeak )
                            spikes_.push_back(std::pair<T,size_t>(t,i));
                }
            }

    public:

            /*! \~russian
             * \brief Конструктор.
             * \details В зависимости от наличия макроса NN_CUDA_IMPL
             * создает имплементацию на CPU или на GPU.
             * \param dNum порядковый номер обозначения данных.
             * \param sNum порядковый номер обозначения модели решателя.
             */
            explicit Solver( const size_t & dNum, const size_t & sNum ) :  
                sNum_(sNum),
                dNum_(dNum),
                pData_
                (
                    Data<T>::createItem(
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

            //! \~russian \brief Деструктор по умолчанию.
            virtual ~Solver() = default;

            //! \~russian \brief Удаленный копирующий конструктор.
            Solver( const Solver& ) = delete;

            //! \~russian \brief Удаленный оператор присваивания.
            Solver& operator=( const Solver& ) = delete;

            //! \~russian \brief Удаленный перемещающий конструктор.
            Solver( const Solver&& ) = delete;

            //! \~russian \brief Удаленный перемещающий оператор присваивания.
            Solver& operator=( const Solver&& ) = delete;

            //! \~russian \brief Перечисление с типами решателей.
            enum ChildId : size_t
            {  
                SolverPCNNI2003EId = 1, //!< \~russian Е.М. Ижикевич, 2003, метод Эйлера
                SolverUNN270117Id  = 2 //!< \~russian Модель ННГУ 27.01.2017, 2017, метод РК.
            };

            /*! \~russian
             * \brief Функция устанавливает флаг формата выводных данных объекта.
             * \details Если входной аргумент равен true, то вывод в поток будет
             * происходить в бинарном формате.
             * \param isbin новый флаг вывода в бинарном виде.
             */
            void setIsBinaryWriteFlag(bool isbin) { isBinaryWrite_ = isbin; pData_->isBinaryWrite_ = isbin; }

            /*! \~russian
             * \brief Функция устанавливает флаг формата входных данных объекта.
             * \details Если входной аргумент равен true, то входные данные в потоке
             * должны быть представлены в бинарном формате.
             * \param isbin новый флаг бинарного формата входных данных.
             */
            void setIsBinaryReadFlag(bool isbin) { isBinaryRead_ = isbin; pData_->isBinaryRead_ = isbin; }

            /*! \~russian
             * \brief Функция возвращает флаг формата выходных данных объекта.
             * \return флаг формата выходных данных объекта. Если true, то выходные данные будут представлены в бинарном виде.
             */
            bool getIsBinaryWriteFlag() const { return isBinaryWrite_; }

            /*! \~russian
             * \brief Функция возвращает флаг формата входных данных для объекта.
             * \return флаг формата входных данных. Если true, то входные данные в потоке должны быть представлены в бинарном виде.
             */
            bool getIsBinaryReadFlag() const { return isBinaryRead_; }

            /*! \~russian
             * \brief Фабричный метод создания конкретного решателя.
             * \param id идентификационный номер решателя.
             * \return уникальный указатель на решатель в соответствии с номером.
             */
            static std::unique_ptr<Solver<T>> createItem( ChildId id )
            {
                std::unique_ptr<Solver<T>> ptr;
                switch( id )
                {
                    case SolverPCNNI2003EId:
                        ptr = std::unique_ptr<Solver<T>>( std::make_unique<SolverPCNNI2003E<T>>() );
                    break;
                    case SolverUNN270117Id:
                    	ptr = std::unique_ptr<Solver<T>>( std::make_unique<SolverUNN270117<T>>() );
                    break;
                }
                return ptr;
            }
            
            /*! \~russian
             * \brief Функция для запуска решателя. Производит все вычисления на заданном отрезке модельного времени.
             * \details Задействует все параметры модели. Производит запуск имплементации решателя,
             * а также записывает прогресс решения (от 0 до 100) в таблицу в базе данных.
             * \param ostr rvalue-ссылка на поток вывода для выходной информации симулятора (.sout).
             * \param dbPtr указатель на объект, который связан с базой данных.
             */
	    void solve( std::ostream && ostr = std::ostream(nullptr) )
            {
            	if ( pData_->dtDump > pData_->tEnd ){
            		pData_->dtDump = pData_->tEnd;
            	}

                T cte = pData_->t;
                if( ostr ) ostr << *this;			// write in outfile

                if ( ! isBinaryWrite_ )
                	 ostr << std::endl;

                cte += pData_->dtDump;
                progress_= 0;

                while( pData_->t < pData_->tEnd ) 
                {
                    solveImpl_( cte );
                    cte = pData_->t;
                    progress_ = round( cte*100./pData_->tEnd );

                    if( ostr ) ostr << *this;		// write in outfile

                    if ( ! isBinaryWrite_ )
                    	 ostr << std::endl;

                    cte += pData_->dtDump;
                } 
                makeSpikes_();
            }

            /*! \~russian
             * \brief Потоковое чтение данных.
             * \details Данная функция используется для чтения данных из входного
             * файла симулятора (.sin) .
             * \param istr ссылка на поток ввода, откуда читаются данные.
             * \return ссылку на поток ввода, откуда были прочитаны данные.
             */
            virtual std::istream& read( std::istream& istr ) = 0 ;

            /*! \~russian
             * \brief Потоковая запись данных.
             * \param ostr ссылка на поток вывода, куда печатаются все параметры и переменные модели.
             * \return ссылку на поток вывода, куда были напечатаны все параметры и переменные модели.
             */
            virtual std::ostream& write( std::ostream& ostr ) const = 0 ;

            /*! \~russian
             * \brief Функция проверяет корректность данных, которые
             * находятся в потоке istr.
             * \details Данные в потоке проверяются на то, подходят ли они
             * для входного файла симулятора.
             * \param istr ссылка на поток, откуда ведется считывание.
             * \return true, если данные подходят для входного файла симулятора.
             */
            virtual std::pair<bool, std::string> checkFile( std::istream& istr ) const = 0;

            /*! \~russian
             * \brief Возвращает итераторы на начало и конец вектора спайков.
             * \return пару из константных итераторов на начало и на конец вектора спайков.
             */
            std::pair<citSpikes,citSpikes> getSpikes() const
            {
                return std::pair<citSpikes,citSpikes>
                    (
                         spikes_.cbegin(),
                         spikes_.cend()
                    );
            }

            /*! \~russian
             * \brief Возвращает итераторы на начало и конец вектора с осцилограммами.
             * \return пару из константных итераторов на начало и конец вектора с осциллограммами напряжения на нейронах.
             */
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

/*! \~russian
 * \brief Оператор потокового вывода.
 * \details Вызывает реализованную внутри класса функцию write().
 * \param ostr ссылка на поток вывода, куда происходит печать информации.
 * \param item ссылка на объект, данные которого необходимо вывести.
 * \return ссылку на поток вывода, куда была напечатана информация.
 */
template<class T>
std::ostream& operator<<( std::ostream & ostr, const NNSimulator::Solver<T> & item)
{
    return (item.write(ostr));
}

/*! \~russian
 * \brief Оператор потокова ввода.
 * \details Вызывает реализованную внутри класса функцию read().
 * \param istr ссылка на поток ввода, откуда происходит считывание данных.
 * \param item объект, куда происходит запись данных из потока ввода.
 * \return ссылку на поток ввода, откуда произошло считывание данных.
 */
template<class T>
std::istream& operator>>( std::istream & istr, NNSimulator::Solver<T> & item)
{
    return (item.read(istr));
}

namespace std{

	/*! \~russian
	 * \brief Перегруженный оператор вывода для пары из unsigned long long int и любого другого типа.
	 * Данную функцию необходимо помещать в пространство имен std,
	 * чтобы можно было воспользоваться коллективной функцией с
	 * итератором вывода.
	 * \param ostr ссылка на поток вывода, куда будут напечатаны данные.
	 * \param p пара, которая должна быть напечатана.
	 * \return ссылку на поток вывода, куда были напечатаны данные.
	 */
    template<class T>
    std::ostream& operator<<( std::ostream & ostr, const std::pair<size_t,T> & p)
    {
        return (ostr<<p.first << ' ' << p.second);
    }

	/*! \~russian
	 * \brief Перегруженный оператор вывода для пары из любого типа и valarray с этим типом.
	 * Данную функцию необходимо помещать в пространство имен std,
	 * чтобы можно было воспользоваться коллективной функцией с
	 * итератором вывода.
	 * \param ostr ссылка на поток вывода, куда будут напечатаны данные.
	 * \param p пара, которая должна быть напечатана.
	 * \return ссылку на поток вывода, куда были напечатаны данные.
	 */
    template<class T>
    std::ostream& operator<<( std::ostream & ostr, const std::pair<T,std::valarray<T>> & p)
    {
        ostr << p.first << '\t';
        for( const auto & e : p.second )
            ostr << e << ' ';
    }
}

#endif

/*@}*/
