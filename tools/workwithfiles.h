#ifndef WORKWITHFILES_GASPARYANMOSES_27092017
#define WORKWITHFILES_GASPARYANMOSES_27092017

#include <iostream>
#include <string>
#include <regex>
#include <fstream>

/*! \~russian
 * \brief Функция проверяет, является ли входной аргумент числом.
 * \param arg параметр, который необходимо проверить.
 * \return true, если это целое число >= 0.
 */
template<typename T>
inline bool isNumber(T arg){
    std::stringstream sstream;
    sstream << arg;
    std::string str;
    sstream >> str;
    std::regex e ("\\d+");
    if (std::regex_match ( str, e ))
        return true;
    else
        return false;
}

/*! \~russian
 * \brief Функция проверяет файл на расширение.
 * \details Проверяет по расширению, является ли файл бинарным или текстовым.
 * Входной параметр extension подразумевается расширением для текстового файла.
 * добавление символа 'b' к нему расценивается как бинарный формат.
 * Функция изменяет данный аргумент на найденное расширение.
 *
 * \param extension расширение файла, которое необходимо обнаружить. После запуска
 * данной функции будет изменено на найденное расширение - заданное или бинарное
 * (.sout или .bsout, .sin или .bsin и т.п.).
 * \param filename имя файла, чье расширение необходимо обнаружить.
 * \param finded флаг, указывающий, найдено ли вообще расширение в имени файла.
 * \return true, если указанный файл имеет расширение бинарного формата, а не
 * текстового (.bsin, а не .sin) .
 */
bool check_file_extension( std::string& extension, const std::string& filename, bool& finded ){
	finded = false;

	std::string oldEx = extension;
	std::string regstr;

	if ( extension.empty() ){
		std::cerr << "ERROR in " << __FUNCTION__
				  << "( std::string& extension, const std::string& filename, bool& finded ) : line "
				  << __LINE__ << ", file " << __FILE__ << std::endl;
		std::cerr << "\targument extension can\'t be an empty string" << std::endl << std::endl;
		throw;
	}

	if ( extension[0] != '.' )
		extension.insert(0, ".");

	regstr = "\\" + extension + "$";

	bool isbin = false;

	std::regex ex( regstr );
	if ( std::regex_search(filename, ex) ){
		finded = true;
		return isbin;
	}
	else{
		extension.insert(1, "b");
		regstr = "\\" + extension + "$";
		ex = std::regex( regstr );

		if ( std::regex_search(filename, ex) ){
			finded = true;
			isbin = true;
		}
		else{
			extension = oldEx;
		}

		return isbin;
	}
}

/*! \~russian
 * \brief Функция проверяет существование файла с именем, указанным в аргументе.
 * \param filename имя файла, который проверяется на существование.
 * \return true, если файл существует и он доступен для чтения.
 */
inline bool testFileExistence(const std::string& filename){
	std::ifstream fin;
	bool result;
	fin.open(filename, std::ios::in);
	result = fin.is_open();
	fin.close();

	return result;
}

/*! \~russian
 * \brief Функция пробует создать файл с именем, указанным в аргументе.
 * Очищает содержимое файла.
 * \param filename имя файла
 * \return true, если есть возможность создать файл с указанным именем.
 */
inline bool testFileCreation(const std::string& filename){
	std::ofstream fout;
	bool result;
	fout.open(filename, std::ios::out);
	result = fout.is_open();
	fout.close();

	return result;
}

/*! \~russian
 * \brief Функция извлекает имя каталога из имени файла.
 * \param[in] filename полное имя файла.
 * \param[out] newInFilename ссылка на строку, куда будет записано короткое имя файла.
 * \return строку, содержащую название директории, в которой находится файл. (заканчивается символом /)
 */
inline std::string getCatalogFromFilename(const std::string& filename, std::string& newInFilename){
	std::string res;
	size_t lastSlash;

	lastSlash = filename.find_last_of("/\\");
	if ( lastSlash != std::string::npos ){
		res = filename.substr(0, lastSlash + 1);
		newInFilename = filename.substr(lastSlash+1);
	}
	else{
		res = "./";
		newInFilename = filename;
	}

	return res;
}

#endif
