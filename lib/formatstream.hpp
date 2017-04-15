/** @addtogroup Plugin
 * @{*/

/** @file */

#ifndef _NNetworkSimulatorFormatStreamNDN2017_
#define _NNetworkSimulatorFormatStreamNDN2017_

#include <iomanip>

//! Класс для форматировия вывода в поток. \details Позволяет настроить ширину поля. Выводит завершающий пробел.
class FormatStream { // FormatOStream

    //! Ширина поля.
    size_t width_;

    //! Поток.
    std::ostream & ostr_;

public:

    //! Конструктор.
    explicit FormatStream( std::ostream & ostr, size_t width = 5 ) :
        ostr_(ostr), width_(width) {}
       
    //! Оператор вывода.
    template<class T> FormatStream& operator<< ( const T& t)
    {
        ostr_ << std::setw(width_) << t << ' ';
        return *this;
    }

    //! Преобразование к потоковому типу.
    operator std::ostream&() { return ostr_; }
};

#endif

/*@}*/
