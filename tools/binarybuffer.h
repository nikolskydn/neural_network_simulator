/*
 * binarybuffer.h
 *
 *  Created on: 21 ����. 2017 �.
 *      Author: Gasparyan Moses
 */
#ifndef BINARYBUFFER_GASPARYANMOSES_21092017
#define BINARYBUFFER_GASPARYANMOSES_21092017

#include <iostream>
#include <vector>
#include <valarray>

#define DEBUGBINARYBUFFER 0

/*! \~russian
 * \brief Бинарный буфер для вывода.
 *
 * Данный класс создан, чтобы с помощью
 * обычного оператора вывода можно было
 * выводить данные в бинарном формате.
 *
 * Поток, в который должен осуществляться вывод,
 * должен быть создан заранее.
 *
 * Схема использования:
 * \code
 *
 * std::string filename = "myBinFile";
 * std::ofstream fout;
 * fout.open(filename, std::ios::out | std::ios::binary);
 *
 * BinaryBufferOutS binbuf(fout);
 *
 * int vi = 34353;
 * double vd = 3434.343434;
 * std::string vs = "Hello!";
 *
 * binbuf << vi << vd << vs;
 *
 * binbuf.flushIt();	// not necessary
 *
 * fout.close();
 *
 * \endcode
 *
 */
class BinaryBufferOutS {
    //! \~russian \brief Поток, в который происходит вывод.
    std::ostream & ostr_;

public:

    /*! \~russian
     * \brief Конструктор.
     * \param ostr ссылка на поток, в который будет происходить вывод.
     */
    explicit BinaryBufferOutS( std::ostream & ostr) : ostr_(ostr) {}

    //! \~russian \brief Деструктор. \details Вызывает функцию flush для потока.
    ~BinaryBufferOutS() { flushIt(); }

    /*! \~russian
     * \brief Оператор вывода, который выводит элемент в виде бинарной информации.
     * \param t произвольный элемент (не си-строка, не строка, не vector и не valarray).
     * \return ссылку на текущий объект.
     */
    template<typename T>
    BinaryBufferOutS& operator<< ( const T& t);

    /*! \~russian
     * \brief Оператор вывода. Перегруженная версия для си-строк.
     * \details Выводит посимвольно всю строку. Не выводит символ конца строки.
     * \param t строка в си-стиле.
     * \return ссылку на текущий объект.
     */
    BinaryBufferOutS& operator<< ( const char* t );

    /*! \~russian
     * \brief Оператор вывода. Перегруженная версия для массива (vector).
     * \details Выводит вектор целым блоком в поток вывода.
     * \param vec константная ссылка на вектор.
     * \return ссылку на текущий объект.
     */
    template<typename T>
    BinaryBufferOutS& operator<< ( const std::vector<T>& vec );

    /*! \~russian
     * \brief Оператор вывода. Перегруженная версия для массива (valarray).
     * \details Выводит массив целым блоком в поток вывода.
     * \param val константная ссылка на массив.
     * \return ссылку на текущий объект.
     */
    template<typename T>
    BinaryBufferOutS& operator<< ( const std::valarray<T>& val );

    /*! \~russian
     * \brief Функция для сброса буфера.
     * \details Т.к. данный класс не является наследником basic_ostream,
     * то std::endl, std::flush нельзя просто так использовать с ним.
     * Поэтому используется данный метод, чтобы произвести сброс буфера.
     */
    void flushIt() { ostr_ << std::flush; }

    /*! \~russian
     * \brief Оператор преобразования в bool.
     * \details Преобразует в false, если bad() или fail()
     * вернули true.
     */
	operator bool(){
		return !( ostr_.fail() || ostr_.bad() );
	}

    /*! \~russian
     * \brief Оператор преобразования в bool. Константная версия.
     * \details Преобразует в false, если bad() или fail()
     * вернули true.
     */
	operator bool() const{
		return !( ostr_.fail() || ostr_.bad() );
	}
};

/*! \~russian \brief Бинарный буфер для ввода.
 *
 * Данный класс создан, чтобы с помощью
 * обычного оператора ввода можно было
 * получить данные из бинарного формата.
 *
 * Поток, из которого должно осуществляться считывание,
 * должен быть создан заранее.
 *
 * Схема использования:
 * \code
 *
 * // writing some information in file
 * std::string filename = "myBinFile";
 * std::ofstream fout;
 * fout.open(filename, std::ios::out | std::ios::binary);
 *
 * BinaryBufferOutS binbuf(fout);
 *
 * int vi = 34353;
 * double vd = 3434.343434;
 * std::string vs = "Hello!";
 *
 * binbuf << vi << vd << vs;
 *
 * binbuf.flushIt();	// not necessary
 *
 * fout.close();
 *
 * // reading from binary file
 * int vi2;
 * double vd2;
 * std::string vs2;
 * vs2.resize( vs.size() );
 *
 * std::ifstream fin;
 * fin.open(filename, std::ios::in | std::ios::binary);
 *
 * BinaryBufferInS binin(fin);
 *
 * binin >> vi >> vd >> vs;
 *
 * std::cout << "vi = " << vi << ", vi2 = " << vi2 << std::endl;
 * std::cout << "vd = " << vd << ", vd2 = " << vd2 << std::endl;
 * std::cout << "vs = " << vs << ", vs2 = " << vs2 << std::endl;
 *
 * fin.close();
 *
 * \endcode
 *
 */
class BinaryBufferInS{
	//! \~russian \brief Поток, из которого происходит считывание.
	std::istream& bf_;
public:
	/*! \~russian
	 * \brief Конструктор.
	 * \param bf ссылка на поток, из которого будет происходить считывание.
	 */
	explicit BinaryBufferInS(std::istream& bf) : bf_(bf) {}

	/*! \~russian
	 * \brief Оператор ввода, который считывает элемент так, будто бы он записан был в бинарном виде.
	 * \param[out] elem ссылка на объект, который будет изменен данной функцией.
	 * \return ссылку на текущий объект-поток.
	 */
	template<typename T>
	BinaryBufferInS& operator>> (T& elem);

	/*! \~russian
	 * \brief Оператор ввода для вектора.
	 * \details Размер вектора должен быть заранее задан. Данная функция считывает ровно
	 * столько байт, сколько требуется для заполнения всего вектора vec.
	 * \param vec ссылка на вектор, который будет записан через данную функцию.
	 * \return ссылку на текущий объект-поток.
	 */
	template<typename T>
	BinaryBufferInS& operator>> (std::vector<T>& vec);

	/*! \~russian
	 * \brief Оператор ввода для valarray.
	 * \details Размер valarray вектора должен быть заранее задан. Данная функция считывает ровно
	 * столько байт, сколько требуется для заполнения всего вектора val.
	 * \param val ссылка на valarray, который будет записан через данную функцию.
	 * \return ссылку на текущий объект-поток.
	 */
	template<typename T>
	BinaryBufferInS& operator>> (std::valarray<T>& val);

	/*! \~russian
	 * \brief Оператор ввода для си-строки.
	 * \details Память под s должна быть выделена заранее.
	 * Символ '\\0' также должен быть поставлен заранее в нужное место, т.к.
	 * считывание происходит до него (strlen()).
	 * \param s указатель на си-строку, которая будет изменена данной функцией.
	 * \return ссылку на текущий объект-поток.
	 */
	BinaryBufferInS& operator>> (char* s);

    /*! \~russian
     * \brief Оператор преобразования в bool.
     * \details Преобразует в false, если bad() или fail()
     * вернули true.
     */
	operator bool(){
		return !( bf_.fail() || bf_.bad() );
	}

    /*! \~russian
     * \brief Оператор преобразования в bool. Константная версия.
     * \details Преобразует в false, если bad() или fail()
     * вернули true.
     */
	operator bool() const{
		return !( bf_.fail() || bf_.bad() );
	}
};

#include "binarybuffer.tcc"

#endif /* BINARYBUFFER_GASPARYANMOSES_21092017 */
