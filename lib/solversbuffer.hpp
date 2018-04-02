/** @addtogroup Plugin
 * @{*/

/** @file */

#ifndef _NNetworkSimulatorSolver_sBufferNDN2017_
#define _NNetworkSimulatorSolver_sBufferNDN2017_

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

/*! \~russian \brief Буфер, пропускающий строки с '#' .
 *
 * Объект класса представляет собой буфер, который
 * считывает всю информацию из файла, пропуская строки с '#' в начале.
 *
 * После считывания файла, объект может быть использован
 * как обычный входной поток.
 *
 * Если файл очень большой (не помещается в оперативную память),
 * то использование данного буфера вызовет segmentation fault,
 * т.к. весь буфер заполняется единожды из файла целиком.
 *
 * Схема использования:
 * \code
 *
 * std::string filename = "someFile.txt";
 *
 * SolversBuffer buff;
 * buff.readFile( filename );
 *
 * int vN;
 * std::string vS;
 * double vD;
 *
 * buff >> vN >> vS >> vD;
 *
 * std::cout << "vN = " << vN << std::endl;
 * std::cout << "vS = " << vS << std::endl;
 * std::cout << "vD = " << vD << std::endl;
 *
 * \endcode
 *
 */
class SolversBuffer : public std::stringstream 
{
	/*! \~russian
	 * \brief Флаг форматирования входных данных.
	 * \details Если true, то берутся просто данные из файла и записываются
	 * в данный строковый буфер.
	 *
	 * Если false, то входной файл рассматривается как текстовый. В таком
	 * случае пропускаются все строки, которые начинаются с символа '\#' .
	 */
	bool getOnlyRawData_{false};

    public:

		/*! \~russian
		 * \brief Функция устанавливает флаг форматирования входных данных.
		 * \param useFlag если true, то входные данные записываются в строковый буфер
		 * без изменений.
		 */
		void useOnlyRawData( bool useFlag ) { getOnlyRawData_ = useFlag; }

		/*! \~russian
		 * \brief Функция возвращает текущий флаг форматирования входных данных.
		 * \return true, если входные данные будут или были записаны в строковый буфер
		 * без изменений. false, если из входного файла были удалены все строки,
		 * начинающиеся с символа '\#' .
		 */
		bool onlyRawDataUsed() const { return getOnlyRawData_; }

		//! \~russian \brief Пустой конструктор.
        explicit SolversBuffer()   {}
    
        /*! \~russian
         * \brief Функция считывает всю информацию из файла в буфер.
         * \details Если такой файл не существует, то выдает ошибку.
         * \param fileName имя файла, откуда происходит считывание.
         */
        void readFile( const std::string & fileName )
        {
           std::ifstream src( fileName );
           if( ! src.is_open() ) {
               std::cerr << "\033[1;31;40merror: \033[1;33;40mopening file '" << fileName << "'\033[0m\n";
               throw;
           }
           if (getOnlyRawData_){
        	   *this << src.rdbuf();
        	   return;
           }

           std::stringstream rawData;
           rawData << src.rdbuf();
           std::string rawLine;
           while( std::getline( rawData, rawLine) )
           {
               if( rawLine[0] != '#' && rawLine[0] != '\n' )
               {
                    *this << rawLine << ' ';
               }
           }
           src.close();
        }

        /*! \~russian
         * \brief Функция очищает поток.
         */
        void clean(){ str(std::string()); }
};
#endif

/*@}*/
