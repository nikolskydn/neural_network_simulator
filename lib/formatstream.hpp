/** @addtogroup Plugin
 * @{*/

/** @file */

#ifndef _NNetworkSimulatorFormatStreamNDN2017_
#define _NNetworkSimulatorFormatStreamNDN2017_

#include <iomanip>

/*! \~russian \brief Класс для форматировия вывода в поток.
 * \details Позволяет настроить ширину поля. Выводит завершающий пробел.
 * Благодаря перегруженному оператору вывода и
 * преобразованию в std::ostream& позволяет использовать
 * объект данного класса как обычный поток вывода.
 */
class FormatStream { // FormatOStream

    //! \~russian \brief Ширина поля.
    size_t width_;

    //! \~russian \brief Ссылка на поток, куда происходит вывод.
    std::ostream & ostr_;

public:

    /*! \~russian
     * \brief Конструктор.
     * \param ostr ссылка на поток, куда происходит вывод.
     * \param width ширина поля.
     */
    explicit FormatStream( std::ostream & ostr, size_t width = 5 ) :
        ostr_(ostr), width_(width) {}
       
    /*! \~russian
     * \brief Оператор вывода.
     * \param t элемент, который будет выведен в поток.
     * \return ссылку на текущий объект.
     */
    template<class T> FormatStream& operator<< ( const T& t)
    {
        ostr_ << std::setw(width_) << t << ' ';
        return *this;
    }

    //! \~russian \brief Преобразование к потоковому типу.
    operator std::ostream&() { return ostr_; }
};

#endif

/*@}*/
