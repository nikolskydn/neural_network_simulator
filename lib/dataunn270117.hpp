/** @addtogroup Data
 * @{*/

/** @file */

#ifndef DATAUNN270117_GASPARYANMOSES_15052017
#define DATAUNN270117_GASPARYANMOSES_15052017

#include <iostream>
#include <sstream>
#include <valarray>
#include <random>
#include <limits>
#include <vector>
#include <fstream>
#include <memory>
#include "../tools/binarybuffer.h"
#include "data.hpp"
#include "puasson.h"
#include "graphinformation.h"

//#define DEBUG 2

namespace NNSimulator{

template<typename T> class Data;
struct GraphInformation;

/*! \~russian \brief Структура с параметрами модели UNN270117, не изменяющимися во времени.
 *
 * Содержит параметры, использованные в
 * дифференциальных уравнениях, предоставленных заказчиком.
 * Данные параметры не изменяются во время выполнения шагов и
 * задаются лишь один раз в программе.
 *
 * Использовать как обычную структуру с открытыми полями.
 *
 */
template<typename T>
struct ConstParamsUNN270117{
	//! \~russian \brief Удельная мембранная емкость. \details Измеряется в 1 мкФ/см^2 .
	T Cm;
	//! \~russian \brief Проводимость ионного натриевого тока. \details Измеряется в мСм/см^2 .
	T g_Na;
	//! \~russian \brief Проводимость ионного калиевого тока. \details Измеряется в мСм/см^2 .
	T g_K;
	//! \~russian \brief Проводимость ионного тока утечки. \details Измеряется в мСм/см^2 .
	T g_leak;
	//! \~russian \brief Постоянный внешний ток. \details Данное значение подается на каждый нейрон.
	T Iapp;

	//! \~russian \brief Реверсивный потенциал натрия.
	T E_Na;
	//! \~russian \brief Реверсивный потенциал калия.
	T E_K;
	//! \~russian \brief Реверсивный потенциал утечки.
	T E_L;

	//! \~russian \brief Реверсивный синаптический потенциал для тормозной связи. \details Для возбуждающей связи Esyn = 0 .
	T Esyn;
	//! \~russian \brief Сдвиг в формуле для синаптического тока.
	T theta_syn;
	//! \~russian \brief Коэффициент в экспоненте в формуле синаптического тока.
	T k_syn;

	//! \~russian \brief Параметр в формуле для потока (I_glu) внешнего вещества (например, глутамата).
	T alphaGlu;
	//! \~russian \brief Параметр в формуле для внеклеточной концентрации нейропередатчика в окрестности нейрона.
	T alphaG;
	//! \~russian \brief Параметр в формуле для внеклеточной концентрации нейропередатчика в окрестности нейрона.
	T bettaG;

	// ------------------- Астроциты

	//! \~russian \brief Количество астроцитов в сети.
	size_t Nastr;

	/*! \~russian
	 * \brief Матрица связи астроцитов с нейронами. Коэффициент регуляции синаптической связи за счет воздействия астроцитов.
	 * Используется при подсчете синаптических токов.
	 * \details В основном матрица единичная, за редкими исключениями.
	 * Матрица имеет размер [Nneurs][Nastr].
	 *
	 * Хранятся все элементы в массиве для экономии места и времени.
	 * Количество элементов в массиве равно [Nneurs x Nastr].
	 * Чтобы получить пару (ji), необходимо использовать
	 * формулу [ i + j*Nastr ], где i - индекс астроцита, j - индекс нейрона.
	 */
	std::valarray<T> wAstrNeurs;

	/*! \~russian
	 * \brief Матрица связи астроцитов между собой в астроцитарной сети.
	 * Необходима для решения уравнения на кальций и на IP3.
	 * \details В матрице по диагонали все 0. Матрица имеет размер [Nastr][Nastr].
	 *
	 * Хранятся все элементы в массиве для экономии места и времени.
	 * Количество элементов в массиве равно [Nastr x Nastr].
	 * Чтобы получить пару (ji), необходимо использовать
	 * формулу [ i + j*Nastr ], где
	 *
	 * i - индекс астроцита, который влияет на j-й ;
	 *
	 * j - индекс астроцита, на которого влияет i-й.
	 */
	std::valarray<bool> astrConns;

	//! \~russian \brief Параметр в дифференциальном уравнении на концентрацию инозитол 1,4,5-трифосфата .
	T tauIP3;
	//! \~russian \brief Параметр в дифференциальном уравнении на концентрацию инозитол 1,4,5-трифосфата . Равновесное значение ИТФ.
	T IP3ast;
	//! \~russian \brief Параметр в дифференциальном уравнении на долю IP3 в эндоплазматическом ретикулуме, которая не была инактивирована кальцием Ca2+ .
	T a2;
	//! \~russian \brief Параметр в дифференциальном уравнении на долю IP3 в эндоплазматическом ретикулуме, которая не была инактивирована кальцием Ca2+ .
	T d1;
	//! \~russian \brief Параметр в дифференциальном уравнении на долю IP3 в эндоплазматическом ретикулуме, которая не была инактивирована кальцием Ca2+ .
	T d2;
	//! \~russian \brief Параметр в дифференциальном уравнении на долю IP3 в эндоплазматическом ретикулуме, которая не была инактивирована кальцием Ca2+ .
	T d3;
	//! \~russian \brief Параметр в формуле для потока (I_channel) кальция из эндоплазматического ретикулума в цитоплазму.
	T d5;

	//! \~russian \brief Параметр в формуле для потока (I_Cadif) разности концентрации кальция между астроцитами.
	T dCa;
	//! \~russian \brief Параметр в формуле для потока (I_IP3dif) разности концентрации иноситола 1,4,5-трифосфата.
	T dIP3;
	//! \~russian \brief Параметр в формуле для потока (I_channel) кальция из эндоплазматического ретикулума в цитоплазму, а также в формуле для пассивного тока (I_leak), соответствующего градиентному переносу кальция через нейтральные каналы мембраны ЭР.
	T c0;
	//! \~russian \brief Параметр в формуле для потока (I_channel) кальция из эндоплазматического ретикулума в цитоплазму, а также в формуле для пассивного тока (I_leak), соответствующего градиентному переносу кальция через нейтральные каналы мембраны ЭР.
	T c1;
	//! \~russian \brief Параметр в формуле для потока (I_channel) кальция из эндоплазматического ретикулума в цитоплазму.
	T v1;
	//! \~russian \brief Параметр в формуле для потока (I_PLC), зависящего от концентрации кальция, в концентрацию ИТФ.
	T v4;
	//! \~russian \brief Параметр в формуле для потока (I_PLC), зависящего от концентрации кальция, в концентрацию ИТФ.
	T alpha;
	//! \~russian \brief Параметр в формуле для потока (I_PLC), зависящего от концентрации кальция, в концентрацию ИТФ.
	T k4;
	//! \~russian \brief Параметр в формуле для пассивного потока (I_leak), соответствующего градиентному переносу кальция через нейтральные каналы мембраны ЭР.
	T v2;
	//! \~russian \brief Параметр в формуле для обратного потока (I_pump) кальция из цитоплазмы в ЭР.
	T v3;
	//! \~russian \brief Параметр в формуле для обратного потока (I_pump) кальция из цитоплазмы в ЭР.
	T k3;
	//! \~russian \brief Параметр в формуле для пассивного потока (I_in) процессов обмена кальция с внешней средой.
	T v5;
	//! \~russian \brief Параметр в формуле для пассивного потока (I_in) процессов обмена кальция с внешней средой.
	T v6;
	//! \~russian \brief Параметр в формуле для пассивного потока (I_in) процессов обмена кальция с внешней средой.
	T k2;
	//! \~russian \brief Параметр в формуле для пассивного потока (I_out) процессов обмена кальция с внешней средой.
	T k1;

};

/*! \~russian \brief Структура с переменными модели UNN270117. Изменяются каждый временной шаг.
 *
 * Содержит переменные, используемые в дифференциальных уравнениях,
 * предоставленных заказчиком.
 * Данные параметры изменяются каждый временной шаг.
 * Вначале программы должны быть заданы начальные значения для параметров.
 * Иначе значения ставятся в 0.
 *
 * Использовать как обычную структуру с открытыми полями.
 *
 */
template<typename T>
struct VariablesUNN270117{
	//! \~russian \brief Активационная натриевая переменная (воротная переменная). \details Размер массива Nneurs.
	std::valarray<T> m;			// V[1]
	//! \~russian \brief Инактивационная натриевая переменная (воротная переменная). \details Размер массива Nneurs.
	std::valarray<T> h;			// V[2]
	//! \~russian \brief Активационная калиевая переменная (воротная переменная). \details Размер массива Nneurs.
	std::valarray<T> n;			// V[3]
	//! \~russian \brief Внеклеточная концентрация нейропередатчика в окрестности нейрона. \details Размер массива Nneurs.
	std::valarray<T> G;

	// ------------------- Астроциты

	//! \~russian \brief Концентрация свободного цитозолического кальция (free cytosolic calcium concentration) на астроците. \details Размер массива Nastr.
	std::valarray<T> Ca;
	//! \~russian \brief Концентрация иноситол 1,4,5-трифосфата (inositol 1,4,5-triphosphate concentration) на астроците. \details Размер массива Nastr.
	std::valarray<T> IP3;

	/*! \~russian
	 * \brief Доля каналов на мембране эндоплазматического ретикулума,
	 * находящихся в открытом состоянии (denotes the fraction of IP3
	 * receptors in endoplasmic reticulum (ER) that have not been inactivated by Ca2+ ).
	 * \details Размер массива Nastr.
	 */
	std::valarray<T> z;

};

/*! \~russian \brief Структура с параметрами генерации тока модели UNN270117.
 *
 * Содержит параметры, используемые для генерации
 * внешнего воздействия на нейроны.
 *
 * Параметр Istim используется в дифференциальных уравнениях,
 * предоставленных заказчиком.
 *
 * Переменная nextTimeEvent должна быть задана ненулевыми
 * значениями вначале, иначе на первом же шаге будет
 * сгенерирован импульс тока на всех нейронах.
 *
 * Использовать как обычную структуру с открытыми полями.
 *
 */
template<typename T>
struct RandomEventGenerator{
	/*! \~russian
	 * \brief Амплитуда тока, возникающего за счет внешнего воздействия по распределению Пуассона.
	 * \details Ток генерируется от -ампл до +ампл .
	 * Генерация тока происходит по распределению Пуассона.
	 */
	T IstimAmplitude;

	/*! \~russian
	 * \brief Частота возникновения события.
	 * \details r - random from 0 to 1;
	 *
	 * \f$ tau = -\frac{1}{frequency} \cdot ln(r) \f$
	 *
	 * nextTime = nextTime + tau;
	 */
	T frequency;

	/*! \~russian
	 * \brief Длительность события в миллисекундах.
	 * \details Т.е. если ток был сгенерирован, то импульс будет
	 * длиться столько миллисекунд.
	 */
	T duration;

	//! \~russian \brief Массив с временем возникновения следующего события (генерации импульса). \details Массив должен быть задан вначале программы, иначе на все нейроны сразу будет подан ток.
	std::valarray<T> nextTimeEvent;
	//! \~russian \brief Текущее значение внешнего воздействия тока на нейроны.
	std::valarray<T> Istim;
};

/*! \~russian \brief Структура со всеми данными модели UNN270117.
 *
 * Структура содержит все необходимые переменные и параметры
 * модели UNN270117.
 *
 * Также есть функции write() и read(), которые позволяют
 * выводить информацию в файл и считывать ее оттуда.
 * Регулярная запись в .sout файл (выходной файл симулятора)
 * происходит с помощью этих функций.
 *
 * Настоятельно рекомендуется не выделять память вручную
 * для массивов, а воспользоваться функцией read().
 * Затем объект можно использовать как обычную структуру
 * с открытыми полями.
 *
 */
template<typename T>
struct DataUNN270117 : public Data<T>{
	using Data<T>::nNeurs;
	using Data<T>::nNeursExc;
	using Data<T>::VNeurs;							// Vm
	using Data<T>::VNeursPeak;
	using Data<T>::VNeursReset;
	using Data<T>::mNeurs;
	using Data<T>::INeurs;							// I
	using Data<T>::wConns;							// wConns
	using Data<T>::t;
	using Data<T>::tEnd;
	using Data<T>::dt;
	using Data<T>::dtDump;
	using Data<T>::isBinaryRead_;
	using Data<T>::isBinaryWrite_;

	//! \~russian \brief Неизменяющиеся во времени параметры модели UNN270117.
	ConstParamsUNN270117<T> constParams;
	//! \~russian \brief Переменные модели UNN270117.
	VariablesUNN270117<T> var;
	//! \~russian \brief Параметры генерации внешнего воздействия на нейроны модели UNN270117.
	RandomEventGenerator<T> randEv;


	//! \~russian \brief Конструктор по умолчанию.
	DataUNN270117() = default;
	//! \~russian \brief Деструктор по умолчанию.
	virtual ~DataUNN270117() = default;

	//! \~russian \brief Конструктор копирования по умолчанию.
	DataUNN270117( const DataUNN270117& otherData ) = default;
	//! \~russian \brief Перемещающий конструктор по умолчанию.
	DataUNN270117( DataUNN270117&& otherData ) = default;
	//! \~russian \brief Оператор присваивания по умолчанию.
	DataUNN270117& operator=( const DataUNN270117& rhsData ) = default;
	//! \~russian \brief Перемещающий оператор присваивания по умолчанию.
	DataUNN270117& operator=( DataUNN270117&& rhsData ) = default;

    /*! \~russian
     * \brief Функция возвращает структуру с необходимой информацией для
     * создания графиков для всех параметров.
     * \details В зависимости от флага бинарного вывода добавляет расширение
     * .animate и .banimate .
     *
     * Также добавляет нижнее подчеркивание вместе с названием параметра.
     * Для модели UNN270117 выводит следующие параметры:
     * V (график и растр), G, m, h, n, Ca, IP3, z, Istim
     * \param mainPartOfName основное имя, к которому прибавляются расширения.
     * \return вектор с параметрами для создания графиков.
     */
    virtual std::vector<typename NNSimulator::GraphInformation> getGraphNames(const std::string& mainPartOfName) const override{
    	std::vector<typename NNSimulator::GraphInformation> res(10);
    	const std::string xDesc = "t, ms";
    	std::string spks, osks;

    	if ( isBinaryWrite_ ){
    		osks = ".boscs";
    		spks = ".bspks";
    	}
    	else{
    		osks = ".oscs";
    		spks = ".spks";
    	}

    	res[0].filename = mainPartOfName + "_V" + spks;
    	res[0].title = "Neurons\' spikes";
    	res[0].xDesc = xDesc;
    	res[0].yDesc = "N";
    	res[0].numberOfGraphics = 1;

    	res[1].filename = mainPartOfName + "_V" + osks;
    	res[1].title = "Voltage on neurons";
    	res[1].xDesc = xDesc;
    	res[1].yDesc = "V";
    	res[1].numberOfGraphics = nNeurs;

    	res[2].filename = mainPartOfName + "_G" + osks;
    	res[2].title = "Concentration of neuromediators";
    	res[2].xDesc = xDesc;
    	res[2].yDesc = "G";
    	res[2].numberOfGraphics = nNeurs;

    	res[3].filename = mainPartOfName + "_m" + osks;
    	res[3].title = "Sodium channel activation";
    	res[3].xDesc = xDesc;
    	res[3].yDesc = "m";
    	res[3].numberOfGraphics = nNeurs;

    	res[4].filename = mainPartOfName + "_h" + osks;
    	res[4].title = "Sodium channel inactivation";
    	res[4].xDesc = xDesc;
    	res[4].yDesc = "h";
    	res[4].numberOfGraphics = nNeurs;

    	res[5].filename = mainPartOfName + "_n" + osks;
    	res[5].title = "Potassium channel activation";
    	res[5].xDesc = xDesc;
    	res[5].yDesc = "n";
    	res[5].numberOfGraphics = nNeurs;

    	res[6].filename = mainPartOfName + "_Ca" + osks;
    	res[6].title = "Free cytosolic calcium concentration";
    	res[6].xDesc = xDesc;
    	res[6].yDesc = "Ca";
    	res[6].numberOfGraphics = constParams.Nastr;

    	res[7].filename = mainPartOfName + "_IP3" + osks;
    	res[7].title = "Inositol 1,4,5-triphosphate concentration";
    	res[7].xDesc = xDesc;
    	res[7].yDesc = "IP3";
    	res[7].numberOfGraphics = constParams.Nastr;

    	res[8].filename = mainPartOfName + "_z" + osks;
    	res[8].title = "Fraction of IP3 receptors in endoplasmic reticulum (ER) that have not been inactivated by Ca2+";
    	res[8].xDesc = xDesc;
    	res[8].yDesc = "z";
    	res[8].numberOfGraphics = constParams.Nastr;
    	return res;
    }

    /*! \~russian
     * \brief Функция преобразует данные в текущей структуре в данные для графика и выводит их.
     * \details Поддерживает бинарный вывод.
     * \param streams ссылка на вектор с указателями на потоки.
     * Все элементы вектора должны располагаться в том же порядке, в каком выдает их функция
     * getGraphNames().
     */
    virtual void writeDataInGraphics( std::vector<std::unique_ptr<std::ofstream>>& streams ) const override{
    	int Nspks = 0;
    	for(size_t i=0; i < nNeurs; ++i){
    		if ( VNeurs[i] >= VNeursPeak )
    			++Nspks;
    	}

    	if ( isBinaryWrite_ ){
    		BinaryBufferOutS boutVspks( *streams[0] ),
							 boutVosks( *streams[1] ),
							 boutGosks( *streams[2] ),
							 boutmosks( *streams[3] ),
							 bouthosks( *streams[4] ),
							 boutnosks( *streams[5] ),
							 boutCaosks( *streams[6] ),
							 boutIP3osks( *streams[7] ),
							 boutzosks( *streams[8] );

    		boutVspks << t << Nspks;
    		boutVosks << t << VNeurs;
    		boutGosks << t << var.G;
    		boutmosks << t << var.m;
    		bouthosks << t << var.h;
    		boutnosks << t << var.n;
    		boutCaosks << t << var.Ca;
    		boutIP3osks << t << var.IP3;
    		boutzosks << t << var.z;
    	}
    	else{
    		*streams[0] << t << ' ' << Nspks << std::endl;

    		writeGraphColumnsInText(*streams[1], t, VNeurs);
    		writeGraphColumnsInText(*streams[2], t, var.G);
    		writeGraphColumnsInText(*streams[3], t, var.m);
    		writeGraphColumnsInText(*streams[4], t, var.h);
    		writeGraphColumnsInText(*streams[5], t, var.n);
    		writeGraphColumnsInText(*streams[6], t, var.Ca);
    		writeGraphColumnsInText(*streams[7], t, var.IP3);
    		writeGraphColumnsInText(*streams[8], t, var.z);
    	}
    }

	/*! \~russian
	 * \brief Функция вывода всех данных модели UNN270117 в поток.
	 * \details Функции write() и read() построены одинаковым образом, т.е.
	 * то, что было выведено с помощью write(), может быть считано с помощью
	 * read() без модификации выведенного текста.
	 * \param ostr ссылка на поток вывода, куда печатаются данные модели UNN270117.
	 * \return ссылку на поток вывода, куда были выведены данные модели UNN270117.
	 */
	virtual std::ostream& write( std::ostream& ostr ) const final override{

		if (isBinaryWrite_){
			#if DEBUG >= 2
			std::cout << "is binary true in dataUNN270117. writing in binary buff..." << std::endl;
			#endif

			BinaryBufferOutS binbuf( ostr );

			// solver
			binbuf << t;
			binbuf << tEnd;
			binbuf << dt;
			binbuf << dtDump;

			// Neurs
			binbuf << nNeurs;
			binbuf << nNeursExc;
			binbuf << VNeursPeak;
			binbuf << VNeursReset;

			// Nastr
			binbuf << constParams.Nastr;

			// constant parameters
			binbuf << constParams.Cm;
			binbuf << constParams.g_Na;
			binbuf << constParams.g_K;
			binbuf << constParams.g_leak;
			binbuf << constParams.Iapp;
			binbuf << constParams.E_Na;
			binbuf << constParams.E_K;
			binbuf << constParams.E_L;
			binbuf << constParams.Esyn;
			binbuf << constParams.theta_syn;
			binbuf << constParams.k_syn;
			binbuf << constParams.alphaGlu;
			binbuf << constParams.alphaG;
			binbuf << constParams.bettaG;

			binbuf << constParams.tauIP3;
			binbuf << constParams.IP3ast;
			binbuf << constParams.a2;
			binbuf << constParams.d1;
			binbuf << constParams.d2;
			binbuf << constParams.d3;
			binbuf << constParams.d5;
			binbuf << constParams.dCa;
			binbuf << constParams.dIP3;
			binbuf << constParams.c0;
			binbuf << constParams.c1;
			binbuf << constParams.v1;
			binbuf << constParams.v4;
			binbuf << constParams.alpha;
			binbuf << constParams.k4;
			binbuf << constParams.v2;
			binbuf << constParams.v3;
			binbuf << constParams.k3;
			binbuf << constParams.v5;
			binbuf << constParams.v6;
			binbuf << constParams.k2;
			binbuf << constParams.k1;

			binbuf << randEv.IstimAmplitude;
			binbuf << randEv.frequency;
			binbuf << randEv.duration;

			// constant matrixes
			binbuf << constParams.wAstrNeurs
				   << constParams.astrConns
				   << wConns
				   << mNeurs;

			// variables
			binbuf << VNeurs
				   << INeurs
				   << var.m
				   << var.h
				   << var.n
				   << var.G
				   << var.Ca
				   << var.IP3
				   << var.z;
		}
		else{
			#if DEBUG >= 2
			std::cout << "is binary false in dataUNN270117. writing in usual buff..." << std::endl;
			#endif
			FormatStream oFStr(ostr);

			// solver
			oFStr << t;
			oFStr << tEnd;
			oFStr << dt;
			oFStr << dtDump;

			// Neurs
			oFStr << nNeurs;
			oFStr << nNeursExc;
			oFStr << VNeursPeak;
			oFStr << VNeursReset;

			// Nastr
			oFStr << constParams.Nastr;

			// constant parameters
			oFStr << constParams.Cm;
			oFStr << constParams.g_Na;
			oFStr << constParams.g_K;
			oFStr << constParams.g_leak;
			oFStr << constParams.Iapp;
			oFStr << constParams.E_Na;
			oFStr << constParams.E_K;
			oFStr << constParams.E_L;
			oFStr << constParams.Esyn;
			oFStr << constParams.theta_syn;
			oFStr << constParams.k_syn;
			oFStr << constParams.alphaGlu;
			oFStr << constParams.alphaG;
			oFStr << constParams.bettaG;

			oFStr << constParams.tauIP3;
			oFStr << constParams.IP3ast;
			oFStr << constParams.a2;
			oFStr << constParams.d1;
			oFStr << constParams.d2;
			oFStr << constParams.d3;
			oFStr << constParams.d5;
			oFStr << constParams.dCa;
			oFStr << constParams.dIP3;
			oFStr << constParams.c0;
			oFStr << constParams.c1;
			oFStr << constParams.v1;
			oFStr << constParams.v4;
			oFStr << constParams.alpha;
			oFStr << constParams.k4;
			oFStr << constParams.v2;
			oFStr << constParams.v3;
			oFStr << constParams.k3;
			oFStr << constParams.v5;
			oFStr << constParams.v6;
			oFStr << constParams.k2;
			oFStr << constParams.k1;

			oFStr << randEv.IstimAmplitude;
			oFStr << randEv.frequency;
			oFStr << randEv.duration;

			// constant matrixes
			for( const auto& e: constParams.wAstrNeurs ) oFStr << e;

			for( const auto& e: constParams.astrConns ) oFStr << e;

			for( const auto& e: wConns ) oFStr << e;

			for( const auto& e: mNeurs ) oFStr << e;

			// variables
			for( const auto& e: VNeurs ) oFStr << e;

			for( const auto& e: INeurs ) oFStr << e;

			for( const auto& e: var.m ) oFStr << e;

			for( const auto& e: var.h ) oFStr << e;

			for( const auto& e: var.n ) oFStr << e;

			for( const auto& e: var.G ) oFStr << e;

			for( const auto& e: var.Ca ) oFStr << e;

			for( const auto& e: var.IP3 ) oFStr << e;

			for( const auto& e: var.z ) oFStr << e;
		}
		#if DEBUG >= 2
		std::cout << "in dataUNN270117. writing in buff complete" << std::endl;
		#endif
		return ostr;
	}


    /*! \~russian
     * \brief Функция записывает в поток данные в читаемом для человека формате.
     * \details Использует специальные метки из текстового редактора клиентской
     * части, чтобы файл можно было просматривать с помощью таблиц и вкладок.
     * \param fout ссылка на поток вывода, куда будут напечатаны данные структуры.
     * \return ссылка на поток вывода, куда были напечатаны данные.
     */
    virtual std::ostream& writeAsSin( std::ostream& fout ) const{
    	const std::string sscalar = "#scalar ";
    	const std::string smatrixname = "#matrixname ";
    	const std::string smatrix = "#matrix ";
    	const std::string svector = "#vector ";
    	const std::string svectorsname = "#vectorsname ";
    	const std::string svectors = "#vectors";

    	fout << sscalar		<< "nS" << std::endl;
    	fout << static_cast<int>( Data<T>::ChildId::DataUNN270117Id ) << std::endl;
    	fout << sscalar		<< "t" << std::endl;
    	fout << t			<< std::endl;
    	fout << sscalar		<< "tEnd" << std::endl;
    	fout << tEnd		<< std::endl;
    	fout << sscalar		<< "dt" << std::endl;
    	fout << dt			<< std::endl;
    	fout << sscalar		<< "dtDump" << std::endl;
    	fout << dtDump		<< std::endl;
    	fout << sscalar		<< "nNeurs" << std::endl;
    	fout << nNeurs		<< std::endl;
    	fout << sscalar		<< "nNeursExc" << std::endl;
    	fout << nNeursExc	<< std::endl;
    	fout << sscalar		<< "VPeak" << std::endl;
    	fout << VNeursPeak	<< std::endl;
    	fout << sscalar		<< "VReset" << std::endl;
    	fout << VNeursReset	<< std::endl;
    	fout << sscalar		<< "Nastr" << std::endl;
    	fout << constParams.Nastr << std::endl << std::endl;

    	fout << sscalar		<< "Cm" 	<< std::endl;
    	fout << constParams.Cm			<< std::endl;
    	fout << sscalar		<< "g_Na" 	<< std::endl;
    	fout << constParams.g_Na		<< std::endl;
    	fout << sscalar		<< "g_K" 	<< std::endl;
    	fout << constParams.g_K 		<< std::endl;
    	fout << sscalar		<< "g_leak" << std::endl;
    	fout << constParams.g_leak		<< std::endl;
    	fout << sscalar		<< "Iapp" 	<< std::endl;
    	fout << constParams.Iapp		<< std::endl;
    	fout << sscalar		<< "E_Na"	<< std::endl;
    	fout << constParams.E_Na		<< std::endl;
    	fout << sscalar		<< "E_K"	<< std::endl;
    	fout << constParams.E_K			<< std::endl;
    	fout << sscalar		<< "E_L"	<< std::endl;
    	fout << constParams.E_L			<< std::endl;
    	fout << sscalar		<< "Esyn"	<< std::endl;
    	fout << constParams.Esyn		<< std::endl;
    	fout << sscalar		<< "thetaSyn" << std::endl;
    	fout << constParams.theta_syn	<< std::endl;
    	fout << sscalar		<< "kSyn"	<< std::endl;
    	fout << constParams.k_syn		<< std::endl;
    	fout << sscalar		<< "alphaGlu" << std::endl;
    	fout << constParams.alphaGlu	<< std::endl;
    	fout << sscalar		<< "alphaG"	<< std::endl;
    	fout << constParams.alphaG		<< std::endl;
    	fout << sscalar		<< "bettaG"	<< std::endl;
    	fout << constParams.bettaG		<< std::endl << std::endl;

    	fout << sscalar		<< "tauIP3"	<< std::endl;
    	fout << constParams.tauIP3		<< std::endl;
    	fout << sscalar		<< "IP3ast"	<< std::endl;
    	fout << constParams.IP3ast		<< std::endl;
    	fout << sscalar		<< "a2"		<< std::endl;
    	fout << constParams.a2			<< std::endl;
    	fout << sscalar		<< "d1"		<< std::endl;
    	fout << constParams.d1			<< std::endl;
    	fout << sscalar		<< "d2"		<< std::endl;
    	fout << constParams.d2			<< std::endl;
    	fout << sscalar		<< "d3"		<< std::endl;
    	fout << constParams.d3			<< std::endl;
    	fout << sscalar		<< "d5"		<< std::endl;
    	fout << constParams.d5			<< std::endl;
    	fout << sscalar		<< "dCa"	<< std::endl;
    	fout << constParams.dCa			<< std::endl;
    	fout << sscalar		<< "dIP3"	<< std::endl;
    	fout << constParams.dIP3		<< std::endl;
    	fout << sscalar		<< "c0"		<< std::endl;
    	fout << constParams.c0			<< std::endl;
    	fout << sscalar		<< "c1"		<< std::endl;
    	fout << constParams.c1			<< std::endl;
    	fout << sscalar		<< "v1"		<< std::endl;
    	fout << constParams.v1			<< std::endl;
    	fout << sscalar		<< "v4"		<< std::endl;
    	fout << constParams.v4			<< std::endl;
    	fout << sscalar		<< "alpha"	<< std::endl;
    	fout << constParams.alpha		<< std::endl;
    	fout << sscalar		<< "k4"		<< std::endl;
    	fout << constParams.k4			<< std::endl;
    	fout << sscalar		<< "v2"		<< std::endl;
    	fout << constParams.v2			<< std::endl;
    	fout << sscalar		<< "v3"		<< std::endl;
    	fout << constParams.v3			<< std::endl;
    	fout << sscalar		<< "k3"		<< std::endl;
    	fout << constParams.k3			<< std::endl;
    	fout << sscalar		<< "v5"		<< std::endl;
    	fout << constParams.v5			<< std::endl;
    	fout << sscalar		<< "v6"		<< std::endl;
    	fout << constParams.v6			<< std::endl;
    	fout << sscalar		<< "k2"		<< std::endl;
    	fout << constParams.k2			<< std::endl;
    	fout << sscalar		<< "k1"		<< std::endl;
    	fout << constParams.k1			<< std::endl << std::endl;

    	fout << sscalar 	<< "IstimAmplitude" << std::endl;
    	fout << randEv.IstimAmplitude	<< std::endl;
    	fout << sscalar		<< "IstimFrequency" << std::endl;
    	fout << randEv.frequency 		<< std::endl;
    	fout << sscalar		<< "IstimDuration"	<< std::endl;
    	fout << randEv.duration 		<< std::endl << std::endl;

    	if ( constParams.Nastr > 0 && nNeurs > 0 ){
    		fout << smatrixname << "wAstrNeurs" << std::endl;
    		fout << smatrix;
    		fout << "a1";
    		for( size_t i=1; i < constParams.Nastr; ++i ){
    			fout << " a" << i+1;
    		}
    		fout << std::endl;

    		Data<T>::printMatrix( constParams.wAstrNeurs, constParams.Nastr, fout );
    	}

    	if ( constParams.Nastr > 0 ){
    		fout << smatrixname << "astrConns" << std::endl;
    		fout << smatrix;
    		fout << '1';
    		for(size_t i=1; i < constParams.Nastr; ++i){
    			fout << ' ' << i+1;
    		}
    		fout << std::endl;

    		Data<T>::printMatrix( constParams.astrConns, constParams.Nastr, fout );
    	}

    	if ( nNeurs > 0 ){
    		fout << smatrixname << "wConns" << std::endl;
    		fout << smatrix;
    		fout << '1';
    		for(size_t i=1; i < nNeurs; ++i){
    			fout << ' ' << i+1;
    		}
    		fout << std::endl;

    		Data<T>::printMatrix( wConns, nNeurs, fout );

    		fout << svector << "spikeMask" << std::endl;
    		Data<T>::printVector( mNeurs, fout );
    		fout << svector << "VNeurs" << std::endl;
    		Data<T>::printVector( VNeurs, fout );
    		fout << svector << "INeurs" << std::endl;
    		Data<T>::printVector( INeurs, fout );
    		fout << svector << "m" << std::endl;
    		Data<T>::printVector( var.m, fout );
    		fout << svector << "h" << std::endl;
    		Data<T>::printVector( var.h, fout );
    		fout << svector << "n" << std::endl;
    		Data<T>::printVector( var.n, fout );
    		fout << svector << "G" << std::endl;
    		Data<T>::printVector( var.G, fout );
    	}

    	if ( constParams.Nastr > 0 ){
    		fout << svector << "Ca" << std::endl;
    		Data<T>::printVector( var.Ca, fout );
    		fout << svector << "IP3" << std::endl;
    		Data<T>::printVector( var.IP3, fout );
    		fout << svector << "z" << std::endl;
    		Data<T>::printVector( var.z, fout );
    	}

    	return fout;
    }

	/*! \~russian
	 * \brief Функция ввода всех данных модели UNN270117 из потока.
	 * \details Функции write() и read() построены одинаковым образом, т.е.
	 * то, что было выведено с помощью write(), может быть считано с помощью
	 * read() без модификации выведенного текста.
	 *
	 * Данная функция также выделяет необходимое место под массивы.
	 *
	 * \param istr ссылка на поток ввода, откуда берутся данные для заполнения структуры.
	 * \return ссылку на поток ввода, откуда были взяты данные.
	 */
	virtual std::istream& read( std::istream& istr ) final override{
		if ( isBinaryRead_ ){
			BinaryBufferInS binin( istr );

			// solver
			binin >> t;
			binin >> tEnd;
			binin >> dt;
			binin >> dtDump;
			// Neurs
			binin >> nNeurs;
			binin >> nNeursExc;
			binin >> VNeursPeak;
			binin >> VNeursReset;

			// Nastr
			binin >> constParams.Nastr;

			// constant parameters
			binin >> constParams.Cm;
			binin >> constParams.g_Na;
			binin >> constParams.g_K;
			binin >> constParams.g_leak;
			binin >> constParams.Iapp;
			binin >> constParams.E_Na;
			binin >> constParams.E_K;
			binin >> constParams.E_L;
			binin >> constParams.Esyn;
			binin >> constParams.theta_syn;
			binin >> constParams.k_syn;
			binin >> constParams.alphaGlu;
			binin >> constParams.alphaG;
			binin >> constParams.bettaG;

			binin >> constParams.tauIP3;
			binin >> constParams.IP3ast;
			binin >> constParams.a2;
			binin >> constParams.d1;
			binin >> constParams.d2;
			binin >> constParams.d3;
			binin >> constParams.d5;
			binin >> constParams.dCa;
			binin >> constParams.dIP3;
			binin >> constParams.c0;
			binin >> constParams.c1;
			binin >> constParams.v1;
			binin >> constParams.v4;
			binin >> constParams.alpha;
			binin >> constParams.k4;
			binin >> constParams.v2;
			binin >> constParams.v3;
			binin >> constParams.k3;
			binin >> constParams.v5;
			binin >> constParams.v6;
			binin >> constParams.k2;
			binin >> constParams.k1;

			binin >> randEv.IstimAmplitude;
			binin >> randEv.frequency;
			binin >> randEv.duration;

			randEv.nextTimeEvent.resize( nNeurs );
			for( auto& e: randEv.nextTimeEvent ) e = 0.0;
			std::mt19937 mtDev;
			std::uniform_real_distribution<T> oneDist(0.0, 1.0);
			for(size_t i=0; i < randEv.nextTimeEvent.size(); ++i){
				RD::createPuassonEvent<T>(randEv.nextTimeEvent[i], mtDev, oneDist, std::numeric_limits<T>::max(), randEv.frequency, randEv.duration);
			}
			randEv.Istim.resize( nNeurs );
			for( auto& e: randEv.Istim ) e = 0.0;

			// resizing all massives
			constParams.wAstrNeurs.resize( nNeurs * constParams.Nastr );
			constParams.astrConns.resize( constParams.Nastr * constParams.Nastr );
			wConns.resize(nNeurs * nNeurs);
			mNeurs.resize(nNeurs);
			VNeurs.resize( nNeurs );
			INeurs.resize( nNeurs );
			var.m.resize( nNeurs );
			var.h.resize( nNeurs );
			var.n.resize( nNeurs );
			var.G.resize( nNeurs );
			var.Ca.resize( constParams.Nastr );
			var.IP3.resize( constParams.Nastr );
			var.z.resize( constParams.Nastr );

			// constant matrixes
			binin >> constParams.wAstrNeurs
				  >> constParams.astrConns
				  >> wConns
				  >> mNeurs;

			// variables
			binin >> VNeurs
				  >> INeurs
				  >> var.m
				  >> var.h
				  >> var.n
				  >> var.G
				  >> var.Ca
				  >> var.IP3
				  >> var.z;
		}
		else{
			// solver
			istr >> t;
			istr >> tEnd;
			istr >> dt;
			istr >> dtDump;
			// Neurs
			istr >> nNeurs;
			istr >> nNeursExc;
			istr >> VNeursPeak;
			istr >> VNeursReset;

			// Nastr
			istr >> constParams.Nastr;

			// constant parameters
			istr >> constParams.Cm;
			istr >> constParams.g_Na;
			istr >> constParams.g_K;
			istr >> constParams.g_leak;
			istr >> constParams.Iapp;
			istr >> constParams.E_Na;
			istr >> constParams.E_K;
			istr >> constParams.E_L;
			istr >> constParams.Esyn;
			istr >> constParams.theta_syn;
			istr >> constParams.k_syn;
			istr >> constParams.alphaGlu;
			istr >> constParams.alphaG;
			istr >> constParams.bettaG;

			istr >> constParams.tauIP3;
			istr >> constParams.IP3ast;
			istr >> constParams.a2;
			istr >> constParams.d1;
			istr >> constParams.d2;
			istr >> constParams.d3;
			istr >> constParams.d5;
			istr >> constParams.dCa;
			istr >> constParams.dIP3;
			istr >> constParams.c0;
			istr >> constParams.c1;
			istr >> constParams.v1;
			istr >> constParams.v4;
			istr >> constParams.alpha;
			istr >> constParams.k4;
			istr >> constParams.v2;
			istr >> constParams.v3;
			istr >> constParams.k3;
			istr >> constParams.v5;
			istr >> constParams.v6;
			istr >> constParams.k2;
			istr >> constParams.k1;

			istr >> randEv.IstimAmplitude;
			istr >> randEv.frequency;
			istr >> randEv.duration;

			randEv.nextTimeEvent.resize( nNeurs );
			for( auto& e: randEv.nextTimeEvent ) e = 0.0;
			std::mt19937 mtDev;
			std::uniform_real_distribution<T> oneDist(0.0, 1.0);
			for(size_t i=0; i < randEv.nextTimeEvent.size(); ++i){
				RD::createPuassonEvent<T>(randEv.nextTimeEvent[i], mtDev, oneDist, std::numeric_limits<T>::max(), randEv.frequency, randEv.duration);
			}
			randEv.Istim.resize( nNeurs );
			for( auto& e: randEv.Istim ) e = 0.0;

			// constant matrixes
			constParams.wAstrNeurs.resize( nNeurs * constParams.Nastr );
			for( auto& e: constParams.wAstrNeurs ) istr >> e;

			constParams.astrConns.resize( constParams.Nastr * constParams.Nastr );
			for( auto& e: constParams.astrConns ) istr >> e;

			wConns.resize(nNeurs * nNeurs);
			for( auto& e: wConns ) istr >> e;

			mNeurs.resize(nNeurs);
			for( auto& e: mNeurs ) istr >> e;

			// variables
			VNeurs.resize( nNeurs );
			for( auto& e: VNeurs ) istr >> e;

			INeurs.resize( nNeurs );
			for( auto& e: INeurs ) istr >> e;

			var.m.resize( nNeurs );
			for( auto& e: var.m ) istr >> e;

			var.h.resize( nNeurs );
			for( auto& e: var.h ) istr >> e;

			var.n.resize( nNeurs );
			for( auto& e: var.n ) istr >> e;

			var.G.resize( nNeurs );
			for( auto& e: var.G ) istr >> e;

			var.Ca.resize( constParams.Nastr );
			for( auto& e: var.Ca ) istr >> e;

			var.IP3.resize( constParams.Nastr );
			for( auto& e: var.IP3 ) istr >> e;

			var.z.resize( constParams.Nastr );
			for( auto& e: var.z ) istr >> e;
		}

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
    	using mess = std::pair<bool, std::string>;
    	T vfloat;
    	bool vbool;
    	size_t vST;

    	auto toStr = [](const T& elem) -> std::string{
    		std::stringstream vss;
    		vss << elem;
    		return vss.str();
    	};

		#ifndef READPARAMMACROSES
		#define READPARAMMACROSES
			#define READPARAM(p, s) if ( !(istr >> p) ) \
				return mess(false, std::string("Small amount of parameters. Can\'t read ") + s);
			#define READPARAMBIN(p, s) if ( !(binin >> p) ) \
				return mess(false, std::string("Small amount of parameters. Can\'t read ") + s);
		#endif

    	if ( isBinaryRead_ ){
    		BinaryBufferInS binin( istr );

    		T vt,vtEnd,vdt,vdtDump;
    		long long int vnNeurs, vNastr;

    		// solver
    		READPARAMBIN( vt, "t" )
    		READPARAMBIN( vtEnd, "tEnd" )
			READPARAMBIN( vdt, "dt" )
			READPARAMBIN( vdtDump, "dtDump" )

    		if ( vt > vtEnd ){
    			return mess(false, "parameter t must be less than tEnd. Now t = " + toStr(vt) + ", tEnd = " + toStr(vtEnd));
    		}
    		if ( vdt > vdtDump ){
    			return mess(false, "parameter dt must be less than dtDump. Now dt = " + toStr(vdt) + ", dtDump = " + toStr(vdtDump));
    		}
    		if ( vdt <= 0.0 ){
    			return mess(false, "parameter dt must be positive. Now dt = " + toStr(vdt));
    		}
    		if ( vdtDump <= 0.0 ){
    			return mess(false, "parameter dtDump must be positive. Now dtDump = " + toStr(vdtDump));
    		}

    		// Neurs
    		READPARAMBIN( vnNeurs, "nNeurs" )
    		READPARAMBIN( vST, "nNeursExc" )
			READPARAMBIN( vfloat, "VNeursPeak" )
			READPARAMBIN( vfloat, "VNeursReset" )

    		if ( vnNeurs < 0 ){
    			return mess(false, "parameter nNeurs (number of neurons) must be more than 0. Now nNeurs = " + toStr(vnNeurs));
    		}

    		// Nastr
    		READPARAMBIN( vNastr, "Nastr" )

    		if ( vNastr < 0 ){
    			return mess(false, "parameter Nastr (number of astrocytes) must be more than 0. Now Nastr = " + toStr(vNastr));
    		}

    		// constant parameters
    		READPARAMBIN( vfloat, "Cm" ) // Cm
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter Cm mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAMBIN( vfloat, "g_Na" )		// g_Na
    		READPARAMBIN( vfloat, "g_K" )		// g_K
			READPARAMBIN( vfloat, "g_leak" )	// g_leak
			READPARAMBIN( vfloat, "Iapp" )		// Iapp
			READPARAMBIN( vfloat, "E_Na" )		// E_Na
			READPARAMBIN( vfloat, "E_K" )		// E_K
			READPARAMBIN( vfloat, "E_L" )		// E_L
			READPARAMBIN( vfloat, "Esyn" )		// Esyn
			READPARAMBIN( vfloat, "theta_syn" )	// theta_syn

    		READPARAMBIN( vfloat, "k_syn" )		// k_syn
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter k_syn mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAMBIN( vfloat, "alphaGlu" )	// alphaGlu
    		READPARAMBIN( vfloat, "alphaG" )	// alphaG
			READPARAMBIN( vfloat, "bettaG" )	// bettaG

    		READPARAMBIN( vfloat, "tauIP3" )	// tauIP3
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter tauIP3 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAMBIN( vfloat, "IP3astr" )	// IP3astr
    		READPARAMBIN( vfloat, "a2" )		// a2
			READPARAMBIN( vfloat, "d1" )		// d1
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter d1 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAMBIN( vfloat, "d2" )		// d2
    		READPARAMBIN( vfloat, "d3" )		// d3
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter d3 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAMBIN( vfloat, "d5" )		// d5
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter d5 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAMBIN( vfloat, "dCa" )		// dCa
    		READPARAMBIN( vfloat, "dIP3" )		// dIP3
			READPARAMBIN( vfloat, "c0" )		// c0
			READPARAMBIN( vfloat, "c1" )		// c1
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter c1 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAMBIN( vfloat, "v1" )		// v1
    		READPARAMBIN( vfloat, "v4" )		// v4
			READPARAMBIN( vfloat, "alpha" )		// alpha
			READPARAMBIN( vfloat, "k4" )		// k4
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter k4 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAMBIN( vfloat, "v2" )		// v2
    		READPARAMBIN( vfloat, "v3" )		// v3
			READPARAMBIN( vfloat, "k3" )		// k3
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter k3 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAMBIN( vfloat, "v5" )		// v5
    		READPARAMBIN( vfloat, "v6" )		// v6
			READPARAMBIN( vfloat, "k2" )		// k2
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter k2 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAMBIN( vfloat, "k1" )		// k1

    		// RANDOM
    		READPARAMBIN( vfloat, "IstimAmplitude" )// IstimAmplitude
    		if ( vfloat < 0.0 )
    			return mess(false, "parameter IstimAmplitude mustn\'t be less than 0. Now IstimAmplitude = " + toStr(vfloat));

    		READPARAMBIN( vfloat, "IstimFrequency" )// IstimFrequency
    		if ( vfloat <= 0.0 )
    			return mess(false, "parameter IstimFrequency must be positive. Now IstimFrequency = " + toStr(vfloat));

    		READPARAMBIN( vfloat, "IstimDuration" )// IstimDuration
    		if ( vfloat <= 0.0 )
    			return mess(false, "parameter IstimDuration must be positive. Now IstimDuration = " + toStr(vfloat));

    		// CONSTANT MATRIXES
    		for( long long int i = 0; i < vnNeurs*vNastr; ++i ){	// wAstrNeurs
    			READPARAMBIN( vfloat, "wAstrNeurs["+toStr(i)+"]" )
    			if ( vfloat < 0.0 )
    				return mess(false, "all parameters from matrix wAstrNeurs must be more or equal to 0. Now element with i = " + toStr(i) + " has a value = " + toStr(vfloat));
    		}
    		// astrConns
    		for( long long int i=0; i < vNastr*vNastr; ++i ) READPARAMBIN( vbool, "astrConns["+toStr(i)+"]" )
    		// wConns
    		for( long long int i=0; i < vnNeurs*vnNeurs; ++i ) READPARAMBIN( vfloat, "wConns["+toStr(i)+"]" )
			// mNeurs
			for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vbool, "mNeurs["+toStr(i)+"]" )

			// VARIABLES
			// VNeurs
			for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vfloat, "VNeurs["+toStr(i)+"]" )
			// INeurs
			for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vfloat, "INeurs["+toStr(i)+"]" )
			// m
			for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vfloat, "m["+toStr(i)+"]" )
			// h
			for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vfloat, "h["+toStr(i)+"]" )
			// n
			for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vfloat, "n["+toStr(i)+"]" )
			// G
			for( long long int i=0; i < vnNeurs; ++i ) READPARAMBIN( vfloat, "G["+toStr(i)+"]" )
			// Ca
			for( long long int i=0; i < vNastr; ++i ) READPARAMBIN( vfloat, "Ca["+toStr(i)+"]" )
			// IP3
			for( long long int i=0; i < vNastr; ++i ) READPARAMBIN( vfloat, "IP3["+toStr(i)+"]" )
			// z
			for( long long int i=0; i < vNastr; ++i ) READPARAMBIN( vfloat, "z["+toStr(i)+"]" )
    	}
    	else{
    		T vt,vtEnd,vdt,vdtDump;
    		long long int vnNeurs, vNastr;

    		// solver
    		READPARAM( vt, "t" )
    		READPARAM( vtEnd, "tEnd" )
			READPARAM( vdt, "dt" )
			READPARAM( vdtDump, "dtDump" )

    		if ( vt > vtEnd ){
    			return mess(false, "parameter t must be less than tEnd. Now t = " + toStr(vt) + ", tEnd = " + toStr(vtEnd));
    		}
    		if ( vdt > vdtDump ){
    			return mess(false, "parameter dt must be less than dtDump. Now dt = " + toStr(vdt) + ", dtDump = " + toStr(vdtDump));
    		}
    		if ( vdt <= 0.0 ){
    			return mess(false, "parameter dt must be positive. Now dt = " + toStr(vdt));
    		}
    		if ( vdtDump <= 0.0 ){
    			return mess(false, "parameter dtDump must be positive. Now dtDump = " + toStr(vdtDump));
    		}

    		// Neurs
    		READPARAM( vnNeurs, "nNeurs" )
    		READPARAM( vST, "nNeursExc" )
			READPARAM( vfloat, "VNeursPeak" )
			READPARAM( vfloat, "VNeursReset" )

    		if ( vnNeurs < 0 ){
    			return mess(false, "parameter nNeurs (number of neurons) must be more than 0. Now nNeurs = " + toStr(vnNeurs));
    		}

    		// Nastr
    		READPARAM( vNastr, "Nastr" )

    		if ( vNastr < 0 ){
    			return mess(false, "parameter Nastr (number of astrocytes) must be more than 0. Now Nastr = " + toStr(vNastr));
    		}

    		// constant parameters
    		READPARAM( vfloat, "Cm" ) // Cm
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter Cm mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAM( vfloat, "g_Na" )		// g_Na
    		READPARAM( vfloat, "g_K" )		// g_K
			READPARAM( vfloat, "g_leak" )	// g_leak
			READPARAM( vfloat, "Iapp" )		// Iapp
			READPARAM( vfloat, "E_Na" )		// E_Na
			READPARAM( vfloat, "E_K" )		// E_K
			READPARAM( vfloat, "E_L" )		// E_L
			READPARAM( vfloat, "Esyn" )		// Esyn
			READPARAM( vfloat, "theta_syn" )	// theta_syn

    		READPARAM( vfloat, "k_syn" )		// k_syn
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter k_syn mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAM( vfloat, "alphaGlu" )	// alphaGlu
    		READPARAM( vfloat, "alphaG" )	// alphaG
			READPARAM( vfloat, "bettaG" )	// bettaG

    		READPARAM( vfloat, "tauIP3" )	// tauIP3
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter tauIP3 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAM( vfloat, "IP3astr" )	// IP3astr
    		READPARAM( vfloat, "a2" )		// a2
			READPARAM( vfloat, "d1" )		// d1
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter d1 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAM( vfloat, "d2" )		// d2
    		READPARAM( vfloat, "d3" )		// d3
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter d3 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAM( vfloat, "d5" )		// d5
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter d5 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAM( vfloat, "dCa" )		// dCa
    		READPARAM( vfloat, "dIP3" )		// dIP3
			READPARAM( vfloat, "c0" )		// c0
			READPARAM( vfloat, "c1" )		// c1
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter c1 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAM( vfloat, "v1" )		// v1
    		READPARAM( vfloat, "v4" )		// v4
			READPARAM( vfloat, "alpha" )		// alpha
			READPARAM( vfloat, "k4" )		// k4
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter k4 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAM( vfloat, "v2" )		// v2
    		READPARAM( vfloat, "v3" )		// v3
			READPARAM( vfloat, "k3" )		// k3
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter k3 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAM( vfloat, "v5" )		// v5
    		READPARAM( vfloat, "v6" )		// v6
			READPARAM( vfloat, "k2" )		// k2
    		if ( vfloat == 0.0 )
    			return mess(false, "parameter k2 mustn\'t be equal to 0. Now it is equal to 0");

    		READPARAM( vfloat, "k1" )		// k1

    		// RANDOM
    		READPARAM( vfloat, "IstimAmplitude" )// IstimAmplitude
    		if ( vfloat < 0.0 )
    			return mess(false, "parameter IstimAmplitude mustn\'t be less than 0. Now IstimAmplitude = " + toStr(vfloat));

    		READPARAM( vfloat, "IstimFrequency" )// IstimFrequency
    		if ( vfloat <= 0.0 )
    			return mess(false, "parameter IstimFrequency must be positive. Now IstimFrequency = " + toStr(vfloat));

    		READPARAM( vfloat, "IstimDuration" )// IstimDuration
    		if ( vfloat <= 0.0 )
    			return mess(false, "parameter IstimDuration must be positive. Now IstimDuration = " + toStr(vfloat));

    		// CONSTANT MATRIXES
    		for( long long int i = 0; i < vnNeurs*vNastr; ++i ){	// wAstrNeurs
    			READPARAM( vfloat, "wAstrNeurs["+toStr(i)+"]" )
    			if ( vfloat < 0.0 )
    				return mess(false, "all parameters from matrix wAstrNeurs must be more or equal to 0. Now element with i = " + toStr(i) + " has a value = " + toStr(vfloat));
    		}
    		// astrConns
    		for( long long int i=0; i < vNastr*vNastr; ++i ) READPARAM( vbool, "astrConns["+toStr(i)+"]" )
    		// wConns
    		for( long long int i=0; i < vnNeurs*vnNeurs; ++i ) READPARAM( vfloat, "wConns["+toStr(i)+"]" )
			// mNeurs
			for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vbool, "mNeurs["+toStr(i)+"]" )

			// VARIABLES
			// VNeurs
			for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vfloat, "VNeurs["+toStr(i)+"]" )
			// INeurs
			for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vfloat, "INeurs["+toStr(i)+"]" )
			// m
			for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vfloat, "m["+toStr(i)+"]" )
			// h
			for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vfloat, "h["+toStr(i)+"]" )
			// n
			for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vfloat, "n["+toStr(i)+"]" )
			// G
			for( long long int i=0; i < vnNeurs; ++i ) READPARAM( vfloat, "G["+toStr(i)+"]" )
			// Ca
			for( long long int i=0; i < vNastr; ++i ) READPARAM( vfloat, "Ca["+toStr(i)+"]" )
			// IP3
			for( long long int i=0; i < vNastr; ++i ) READPARAM( vfloat, "IP3["+toStr(i)+"]" )
			// z
			for( long long int i=0; i < vNastr; ++i ) READPARAM( vfloat, "z["+toStr(i)+"]" )
    	}
		#undef READPARAM
		#undef READPARAMBIN
		#undef READPARAMMACROSES
    	return mess(true, "");
    }

	/*! \~russian
	 * \brief Функция для сравнения 2-х параметров.
	 * \details Печатает информацию, что элементы не одинаковы, если это так.
	 * \param lhs первое значение.
	 * \param rhs второе значение.
	 * \param name имя переменной.
	 * \return true, если элементы равны.
	 */
	template<typename CmpT>
	static bool compareIt( const CmpT& lhs, const CmpT& rhs, const std::string& name ) {
		if ( lhs != rhs ){
			#if DEBUG >= 2
			std::cout << "Parameter " << name << " are not equal in two structures" << std::endl;
			std::cout << "   lhs = " << lhs << ", rhs = " << rhs << std::endl << std::endl;
			#endif
			return false;
		}
		else{
			return true;
		}
	}

	/*! \~russian
	 * \brief Функция для сравнения 2-х векторов.
	 * \details Печатает информацию о векторах, если элементы не одинаковы.
	 * \param lhs первое значение.
	 * \param rhs второе значение.
	 * \param name имя переменной.
	 * \return true, если элементы равны.
	 */
	template<typename CmpT>
	static bool compareVal( const std::valarray<CmpT>& lhs, const std::valarray<CmpT>& rhs, const std::string& name ){
		bool res = true;

		if ( lhs.size() != rhs.size() ){
			#if DEBUG >= 2
			std::cout << "Valarrays "
					<< name
					<< " are not equal in two structures" << std::endl;
			std::cout <<
					"   lhs.size() = "
					<< lhs.size() <<
					", rhs.size() = " << rhs.size()
					<< std::endl << std::endl;
			#endif
			return false;
		}

		for( size_t i=0; i < lhs.size(); ++i ){
			if ( rhs[i] != lhs[i] ){
				#if DEBUG >= 2
				std::cout << "Valarrays "
						<< name
						<< " are not equal in two structures" << std::endl;
				std::cout << "element "<<i<<" is differ:" << std::endl;
				std::cout << "   lhs["<<i<<"] = "<<lhs[i]<<", rhs["<<i<<"] = "<<rhs[i] << std::endl;
				#endif
				res = false;
				break;
			}
		}

		return res;
	}

	/*! \~russian
	 * \brief Макрос для сравнения двух величин.
	 * \details Вызывает функцию для сравнения двух величин, но
	 * при это требует только один параметр вместо 3-х.
	 * При неравенстве параметров сразу возвращает false,
	 * что приводит к выходу из функции ==.
	 * \param v название величины, которая является полем структуры.
	 * Также параметр преобразуется в строку для вывода в случае, если
	 * параметры не равны.
	 * \return false, если параметры не равны.
	 */
	#define CMPR(v) if ( ! compareIt(v, oth.v, #v) ) return false;
	/*! \~russian
	 * \brief Макрос для сравнения двух векторов.
	 * \details Вызывает функцию для сравнения двух valarray, но
	 * при это требует только один параметр вместо 3-х.
	 * При неравенстве параметров сразу возвращает false,
	 * что приводит к выходу из функции ==.
	 * \param v название величины, которая является полем структуры.
	 * Также параметр преобразуется в строку для вывода в случае, если
	 * параметры не равны.
	 * \return false, если параметры не равны.
	 */
	#define CMPRVAL(v) if ( ! compareVal(v, oth.v, #v) ) return false;

	/*! \~russian
	 * \brief Перегруженный оператор ==.
	 * \param oth константная ссылка на структуру справа от оператора.
	 * \return true, если поля структур полностью совпадают.
	 */
	bool operator==( const DataUNN270117<T>& oth ) const {
		bool res = true;

		CMPR(t)
		CMPR(dt)
		CMPR(tEnd)
		CMPR(dtDump)
		CMPR(nNeurs)
		CMPR(nNeursExc)
		CMPR(VNeursPeak)
		CMPR(VNeursReset)
		CMPR(constParams.Nastr)

		CMPR(constParams.Cm)
		CMPR(constParams.g_Na)
		CMPR(constParams.g_K)
		CMPR(constParams.g_leak)
		CMPR(constParams.Iapp)
		CMPR(constParams.E_Na)
		CMPR(constParams.E_K)
		CMPR(constParams.E_L)
		CMPR(constParams.Esyn)
		CMPR(constParams.theta_syn)
		CMPR(constParams.k_syn)
		CMPR(constParams.alphaGlu)
		CMPR(constParams.alphaG)
		CMPR(constParams.bettaG)

		CMPR(constParams.tauIP3)
		CMPR(constParams.IP3ast)
		CMPR(constParams.a2)
		CMPR(constParams.d1)
		CMPR(constParams.d2)
		CMPR(constParams.d3)
		CMPR(constParams.d5)
		CMPR(constParams.dCa)
		CMPR(constParams.dIP3)
		CMPR(constParams.c0)
		CMPR(constParams.c1)
		CMPR(constParams.v1)
		CMPR(constParams.v4)
		CMPR(constParams.alpha)
		CMPR(constParams.k4)
		CMPR(constParams.v2)
		CMPR(constParams.v3)
		CMPR(constParams.k3)
		CMPR(constParams.v5)
		CMPR(constParams.v6)
		CMPR(constParams.k2)
		CMPR(constParams.k1)

		CMPR(randEv.IstimAmplitude)
		CMPR(randEv.frequency)
		CMPR(randEv.duration)

		CMPRVAL(constParams.wAstrNeurs)
		CMPRVAL(constParams.astrConns)
		CMPRVAL(wConns)
		CMPRVAL(mNeurs)

		CMPRVAL(VNeurs)
		CMPRVAL(INeurs)
		CMPRVAL(var.m)
		CMPRVAL(var.h)
		CMPRVAL(var.n)
		CMPRVAL(var.G)
		CMPRVAL(var.Ca)
		CMPRVAL(var.IP3)
		CMPRVAL(var.z)

		return res;
	}

	#undef CMPR
	#undef CMPRVAL
};


} // namespace


#endif // DATAUNN270117_GASPARYANMOSES_15052017

/*@}*/
