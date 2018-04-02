#ifndef SYSTEMTIMESTOPWATCH_GASPARYANMOSES_18102017
#define SYSTEMTIMESTOPWATCH_GASPARYANMOSES_18102017

#include "istopwatch.h"
#include <chrono>
#include <ctime>

/*! \~russian \brief Секундомер на основе C++ библиотеки chrono.
 *
 * Использует библиотеку chrono из С++, что позволяет
 * получить точную разницу во времени, т.к. обращение
 * осуществляется к системным часам.
 *
 * Схема использования:
 * \code
 *
 * SystemTimeStopwatch sts;
 * sts.start();
 *
 * std::this_thread::sleep_for( std::chrono::seconds(5) );
 *
 * auto diff = sts.lookTimeSegmentFromStart();
 * std::cout << "diff = " << diff << std::endl;
 *
 * \endcode
 *
 */
class SystemTimeStopwatch : public IStopwatch<std::chrono::duration<double>>{
public:
	//! \~russian \brief Тип переменной, принимающей значение от функции nowTime().
	using TimePoint = std::chrono::high_resolution_clock::time_point;
private:

	//! \~russian \brief Время начала.
	TimePoint start_;
	//! \~russian \brief Время, когда последний раз было просмотрена разность времен с помощью функции lookTimeSegmentFromLast().
	TimePoint lastLooked_;

	/*! \~russian
	 * \brief Функция, возвращающая текущее время.
	 * \return текущее время.
	 */
	inline static TimePoint nowTime() { return std::chrono::high_resolution_clock::now(); }

public:
	//! \~russian \brief Конструктор. \details Ставит значение start_ на время создания объекта.
	SystemTimeStopwatch()
      : IStopwatch<std::chrono::duration<double>>(), start_( nowTime() ), lastLooked_(start_) {}

	/*! \~russian
	 * \brief Функция возвращает значение времени начала.
	 * \return значение времени начала.
	 */
	TimePoint getStartTime() const { return start_; }

	/*! \~russian
	 * \brief Функция возвращает значение времени последнего просмотра.
	 * \return значение времени последнего просмотра (сегмента).
	 */
	TimePoint getLastSegmentStartTime() const { return lastLooked_; }

	/*! \~russian
	 * \brief Функция запуска таймера.
	 * \details При повторном вызове перезапускает секундомер.
	 */
	virtual void start() override;

	/*! \~russian
	 * \brief Функция смотрит время, прошедшее с последнего просмотра с помощью этой функции.
	 * \details В первый раз показывает время от запуска секундомера.
	 * \return время, прошедшее с последнего просмотра.
	 */
	virtual PrintTimeType lookTimeSegmentFromLast() override;

	/*! \~russian
	 * \brief Функция возвращает время, прошедшее с запуска секундомера.
	 * \return время, прошедшее с запуска секундомера.
	 */
	virtual PrintTimeType lookTimeSegmentFromStart() override;
};

// ---------------------- realization

inline void SystemTimeStopwatch::start(){
	start_ = nowTime();
	lastLooked_ = start_;
}

inline SystemTimeStopwatch::PrintTimeType SystemTimeStopwatch::lookTimeSegmentFromLast() {
	auto tmp = nowTime();
	auto diff = tmp - lastLooked_;
	lastLooked_ = tmp;
	return diff;
}

inline SystemTimeStopwatch::PrintTimeType SystemTimeStopwatch::lookTimeSegmentFromStart() {
	return nowTime() - start_;
}

#endif // SYSTEMTIMESTOPWATCH_GASPARYANMOSES_18102017
