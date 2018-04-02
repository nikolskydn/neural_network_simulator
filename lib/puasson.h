/** @addtogroup Plugin
 * @{*/

/** @file */

#ifndef PUASSON_DISTRIBUTION_GASPARYANMOSES_14062017
#define PUASSON_DISTRIBUTION_GASPARYANMOSES_14062017

#include <random>
#include <cmath>

    namespace RD{

    	/*! \~russian
    	 * \brief Функция генерирует время следующего события по распределению Пуассона.
    	 * \details Если текущее время (timeNow_in) выше, чем время
    	 * следующего события + время длительности события
    	 * (nextTimeEvent_in + duration_in) - это означает, что событие
    	 * уже произошло и завершилось, то происходит генерация
    	 * следующего времени для события.
    	 *
    	 * Отрезок времени, который необходимо прибавить к nextTimeEvent_in,
    	 * высчитывается по следующей формуле:
    	 *
    	 * \f$ tau = -\frac{1}{frequency} \cdot ln(gen) \f$
    	 *
    	 * где gen - случайное число от 0 до 1.
    	 *
    	 * \param[out] nextTimeEvent_in ссылка на время следующего события. Значение может быть изменено.
    	 * \param[out] mtDev_in вихрь Мерсена - генератор случайного числа.
    	 * \param[out] oneDist_in линейное распределение от 0 до 1. Используется для генерации случайного числа.
    	 * \param[in] timeNow_in значение, обозначающее текущее время.
    	 * \param[in] frequency_in частота возникновения события.
    	 * \param[in] duration_in длительность возникающих событий.
    	 */
        template<typename T>
        inline void createPuassonEvent(
	        T& nextTimeEvent_in,
    	    std::mt19937& mtDev_in,
    	    std::uniform_real_distribution<T>& oneDist_in,
	        const T timeNow_in,
        	const T frequency_in,
	        const T duration_in
        )
        {
	        if ( timeNow_in >= (nextTimeEvent_in + duration_in) ){
		       T tau = -1.0/frequency_in * log(oneDist_in(mtDev_in));
    		    nextTimeEvent_in += tau;
	        }
        }
    }

#endif // PUASSON_DISTRIBUTION_GASPARYANMOSES_14062017

/*@}*/
