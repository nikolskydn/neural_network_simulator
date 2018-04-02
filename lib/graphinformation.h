/** @addtogroup Data
 * @{*/

/** @file */

#ifndef GRAPH_INFORMATION_GASPARYANMOSES_09102017
#define GRAPH_INFORMATION_GASPARYANMOSES_09102017

#include <string>
#include <iostream>

namespace NNSimulator{

/*! \~russian \brief Структура с необходимой информацией по графику.
 *
 * Используется в методах, возвращающих всю необходимую информацию обо
 * всех графиках, кроме координат точек.
 */
struct GraphInformation{
	//! \~russian \brief Имя файла, содержащего данные графика.
	std::string filename {""};
	//! \~russian \brief Описание графика. Показывается, в основном, сверху графика.
	std::string title {""};
	//! \~russian \brief Подпись для оси X.
	std::string xDesc {""};
	//! \~russian \brief Подпись для оси Y.
	std::string yDesc {""};
	//! \~russian \brief Количество графиков \details Равно количеству нейронов или количеству астроцитов.
	unsigned int numberOfGraphics {0};

	//! \~russian \brief Конструктор по умолчанию.
	GraphInformation() = default;
	//! \~russian \brief Деструктор по умолчанию.
	~GraphInformation() = default;
	//! \~russian \brief Копирующий конструктор по умолчанию.
	GraphInformation(const GraphInformation&) = default;
	//! \~russian \brief Перемещающий конструктор по умолчанию.
	GraphInformation(GraphInformation&&) = default;
};

/*! \~russian
 * \brief Функция печатает в текстовом формате время и параметры.
 * \details Используется из-за отсутствия у контейнеров перегруженного
 * оператора вывода.
 * Печатает время, а затем через разделительный символ все элементы
 * контейнера.
 * В конце добавляет переход на новую строку.
 * \param fout ссылка на поток вывода, куда печатается информация.
 * \param time время, соответствующее срезу параметров.
 * \param vec ссылка на контейнер, содержащий параметры, которые
 * необходимо вывести в файл с графиками.
 */
template<typename TimeType,typename Container>
inline void writeGraphColumnsInText(std::ostream& fout, const TimeType& time, const Container& vec){
	fout << time;
	for(const auto& p: vec)
		fout << ' ' << p;
	fout << std::endl;
}

}

#endif // GRAPH_INFORMATION_GASPARYANMOSES_09102017

/*@}*/
