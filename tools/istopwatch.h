#ifndef ISTOPWATCH_GASPARYANMOSES_18102017
#define ISTOPWATCH_GASPARYANMOSES_18102017

#include <string>
#include <sstream>
#include <chrono>

/*! \~russian \brief Интерфейс для создания секундомеров.
 *
 * Класс представляет собой интерфейс для создания собственных
 * секундомеров.
 *
 * Функция start() запускает секундомер.
 *
 * lookTimeSegmentFromLast() позволяет посмотреть
 * время с последнего просмотра с помощью этой (и только этой) функции.
 *
 * lookTimeSegmentFromStart() позволяет посмотреть
 * время, прошедшее с запуска секундомера.
 *
 */
template<typename T>
class IStopwatch{
public:
	//! \~russian \brief Основной тип времени, который будет возвращен в функциях.
	using PrintTimeType = T;

	//! \~russian \brief Виртуальный деструктор по умолчанию.
	virtual ~IStopwatch() = default;

	/*! \~russian
	 * \brief Функция запуска таймера.
	 */
	virtual void start() = 0;

	/*! \~russian
	 * \brief Функция смотрит время, прошедшее с последнего просмотра с помощью этой функции.
	 * \details В первый раз показывает время от запуска секундомера.
	 * \return время, прошедшее с последнего просмотра.
	 */
	virtual T lookTimeSegmentFromLast() = 0;

	/*! \~russian
	 * \brief Функция возвращает время, прошедшее с запуска секундомера.
	 * \return время, прошедшее с запуска секундомера.
	 */
	virtual T lookTimeSegmentFromStart() = 0;

};

/*! \~russian
 * \brief Функция конвертирует число с секундами в строку.
 * \details Время печатается в формате h m s ms, где
 *
 * h - часы
 *
 * m - минуты
 *
 * s - секунды
 *
 * ms - миллисекунды
 *
 * Если какой-то из параметров равен 0, то он не будет напечатан.
 *
 * \param NsecST количество секунд. Может быть дробным.
 * \return строку со временем.
 */
inline std::string convertTimeIntoString( const double& NsecST ){
	using namespace std::chrono;

	milliseconds Nmillisec( static_cast<size_t>(NsecST * 1000.0) );
	seconds Nsec = duration_cast<seconds>( Nmillisec );
	minutes Nminutes = duration_cast<minutes>( Nmillisec );
	hours Nhours = duration_cast<hours>( Nmillisec );

	milliseconds resMS;
	seconds resSec;
	minutes resMin;
	hours resH;

	resH = Nhours;
	resMin = Nminutes - duration_cast<minutes>( Nhours );
	resSec = Nsec - duration_cast<seconds>( Nminutes );
	resMS = Nmillisec - duration_cast<milliseconds>( Nsec );

	// convert to string
	auto toStr = [](long long int N){
		std::stringstream vss;
		vss << N;
		return vss.str();
	};

	std::string resultStr;
	resultStr = ( resH.count() ? toStr( resH.count() )+" h " : "")
			+ ( resMin.count() ? toStr( resMin.count() )+" m " : "")
			+ ( resSec.count() ? toStr( resSec.count() )+" s " : "")
			+ ( resMS.count() ? toStr( resMS.count() )+" ms" : "");

	if ( resultStr.size() == 0 )
		return "0 ms";

	// deleting spaces in the beginning and in the end
	size_t firstNotSpace;
	size_t lastNotSpace = 0;
	for(size_t i=0; i < resultStr.size(); ++i){
		if ( resultStr[i] != ' ' ){
			firstNotSpace = i;
			break;
		}
	}

	for(int i=resultStr.size() - 1; i >= 0; --i){
		if ( resultStr[i] != ' ' ){
			lastNotSpace = i;
			break;
		}
	}
	resultStr = resultStr.substr( firstNotSpace, lastNotSpace - firstNotSpace + 1 );

	return resultStr;
}

#endif // ISTOPWATCH_GASPARYANMOSES_18102017
