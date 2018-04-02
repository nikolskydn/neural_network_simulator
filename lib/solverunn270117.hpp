/** @addtogroup Solvers
 * @{*/

/** @file */

#ifndef SOLVERUNN270117_GASPARYANMOSES_150517
#define SOLVERUNN270117_GASPARYANMOSES_150517

#include <iostream>
#include <valarray>
#include <sstream>
#include "datacreator.hpp"
#include "solver.hpp"
#include "../tools/binarybuffer.h"

namespace NNSimulator{


template<typename T> class Solver;

/*! \~russian \brief Класс использует численную схему модели UNN270117.
 *
 * Выполняет вызов метода solveUNN270117() из класса SolverImpl.
 *
 */
template<typename T>
class SolverUNN270117 : public Solver<T>{
	using Solver<T>::sNum_;
	using Solver<T>::dNum_;
	using Solver<T>::pImpl_;
	using Solver<T>::pData_;
	using Solver<T>::spikes_;
	using Solver<T>::oscillograms_;
	using Solver<T>::isBinaryWrite_;
	using Solver<T>::isBinaryRead_;

protected:

	//! \~russian \brief Выполнить решение. \details Выполняется вызов установленной реализации.
	virtual void solveImpl_( const T& cte ) final override{
		auto pD = static_cast<DataUNN270117<T>*>(pData_.get());
        pImpl_->solveUNN270117(
        	pD->nNeurs,
			pD->nNeursExc,
			pD->constParams.Nastr,
			pD->VNeursPeak,
			pD->VNeursReset,
			pD->dt,
			cte,
			pD->t,

			pD->constParams.Cm,
			pD->constParams.g_Na,
			pD->constParams.g_K,
			pD->constParams.g_leak,
			pD->constParams.Iapp,
			pD->constParams.E_Na,
			pD->constParams.E_K,
			pD->constParams.E_L,
			pD->constParams.Esyn,
			pD->constParams.theta_syn,
			pD->constParams.k_syn,

			pD->constParams.alphaGlu,
			pD->constParams.alphaG,
			pD->constParams.bettaG,

			pD->constParams.tauIP3,
			pD->constParams.IP3ast,
			pD->constParams.a2,
			pD->constParams.d1,
			pD->constParams.d2,
			pD->constParams.d3,
			pD->constParams.d5,

			pD->constParams.dCa,
			pD->constParams.dIP3,
			pD->constParams.c0,
			pD->constParams.c1,
			pD->constParams.v1,
			pD->constParams.v4,
			pD->constParams.alpha,
			pD->constParams.k4,
			pD->constParams.v2,
			pD->constParams.v3,
			pD->constParams.k3,
			pD->constParams.v5,
			pD->constParams.v6,
			pD->constParams.k2,
			pD->constParams.k1,

			pD->randEv.IstimAmplitude,
			pD->randEv.frequency,
			pD->randEv.duration,
			pD->randEv.nextTimeEvent,
			pD->randEv.Istim,

			pD->wConns,
			pD->constParams.wAstrNeurs,
			pD->constParams.astrConns,

			pD->VNeurs,
			pD->mNeurs,
			pD->INeurs,
			pD->var.m,
			pD->var.h,
			pD->var.n,
			pD->var.G,

			pD->var.Ca,
			pD->var.IP3,
			pD->var.z,

			oscillograms_
        );
	}

public:
	//! \~russian \brief Конструктор.
	explicit SolverUNN270117() : Solver<T>(2,2) {}
	//! \~russian \brief Деструктор по умолчанию.
	virtual ~SolverUNN270117() = default;

	//! \~russian \brief Удаленный копирующий конструктор.
	SolverUNN270117( const SolverUNN270117& ) = delete;
	//! \~russian \brief Удаленный перемещающий конструктор.
	SolverUNN270117( SolverUNN270117&& ) = delete;
	//! \~russian \brief Удаленный оператор присваивания.
	SolverUNN270117& operator=( const SolverUNN270117& ) = delete;
	//! \~russian \brief Удаленный перемещающий оператор присваивания.
	SolverUNN270117& operator=( SolverUNN270117&& ) = delete;

	/*! \~russian
	 * \brief Метод вывода параметров в поток.
	 * \details Функции write() и read() для решателя не симметричны:
	 * функция вывода (write()) сначала печатает номер симулятора,
	 * номер данных, а затем уже сами данные.
	 *
	 * Функция read() выводит только данные.
	 *
	 * \param ostr ссылка на поток вывода, куда печатается информация данного решателя.
	 * \return ссылку на поток вывода, куда была напечатана информация из данного решателя.
	 */
	virtual std::ostream& write( std::ostream& ostr ) const final override{
		auto pD = static_cast<DataUNN270117<T>*>(pData_.get());

		if (isBinaryWrite_){
			#if DEBUG >= 2
				std::cout << "is binary true in SolverUNN270117. writing in binary buff..." << std::endl;
			#endif
			BinaryBufferOutS binbuf( ostr );
			binbuf << sNum_;
			binbuf << dNum_;
		}
		else{
			#if DEBUG >= 2
				std::cout << "is binary false in SolverUNN270117. writing in usual buff..." << std::endl;
			#endif
			FormatStream oFStr( ostr );
			oFStr << sNum_;
			oFStr << dNum_;
		}
		ostr << *pD;
		#if DEBUG >= 2
			std::cout << "in SolverUNN270117. writing in buff complete" << std::endl;
		#endif
		return ostr;
	}

	/*! \~russian
	 * \brief Метод ввода параметров из потока.
	 * \details Функции write() и read() для решателя не симметричны:
	 * функция вывода (write()) сначала печатает номер симулятора,
	 * номер данных, а затем уже сами данные.
	 *
	 * Функция read() выводит только данные.
	 *
	 * \param istr ссылка на поток ввода, откуда берется информация для записи её в решатель.
	 * \return ссылку на поток ввода, откуда была считана информация.
	 */
	virtual std::istream& read( std::istream& istr ) final override{
		auto pD = static_cast<DataUNN270117<T>*>( pData_.get() );
		pD->read(istr);
		return istr;
	}

    /*! \~russian
     * \brief Функция проверяет корректность данных, которые
     * находятся в потоке istr.
     * \details Данные в потоке проверяются на то, подходят ли они
     * для входного файла симулятора.
     * \param istr ссылка на поток, откуда ведется считывание.
     * \return true, если данные подходят для входного файла симулятора.
     */
    virtual std::pair<bool, std::string> checkFile( std::istream& istr ) const{
    	auto pD = static_cast<DataUNN270117<T>*>( pData_.get() );
    	return pD->checkFile( istr );
    }

};


} // namespace

#endif /* SOLVERUNN270117_GASPARYANMOSES_150517 */

/*@}*/
