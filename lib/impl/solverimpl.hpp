/** @addtogroup SolversImpl
 * @{*/

/** @file */

#ifndef _NNetworkSimulatorSolverImplNDN2017_
#define _NNetworkSimulatorSolverImplNDN2017_

#include <valarray>
#include <deque>
#include <memory>

namespace NNSimulator {

    //! \~russian \brief Определяет интерфейс реализаций решателей.
    template<class T> class SolverImpl {
        public:

            /*! \~russian \brief Интерфейс реализации модели Е.М. Ижикевича (2003) методом Эйлера.
             *
             * \details Численная схема:
             * 
             *
             * \f$ V_{j+ \frac 1 2} = V_j +  \frac{\Delta t}{2} \left( 0.04V_j^2+5V_j+140-U_j+I_j \right) \f$, 
             *
             * \f$V_{j+1} = V_{j+\frac 1 2} +  
             * \frac{\Delta t}{2} \left( 0.04V_{j+\frac 1 2}^2+5V_{j+\frac 1 2}+140-U_j+I_j \right) \f$,
             *
             * \f$U_{j+1}=a(bV_{j+1}-U_j)\f$,
             *
             * \f${\rm if} \; V\leq 30 \; mV, \; {\rm then} \; \{ V=V_{Reset} \; and \; U+=U_{Reset} \}.\f$
             *
             * Izhikevich E.M. Simple model of spiking neurons// IEEE transactions of neural networks. V.14. N6. 2003. PP.1569--1572. (http://www.izhikevich.org/publications/spikes.pdf)
             */
            virtual void solvePCNNI2003E
            (
                const size_t & nNeurs,
                const size_t & nNeursExc,
                const T & VNeursPeak,
                const std::valarray<T> & aNeurs,
                const std::valarray<T> & bNeurs,
                const std::valarray<T> & cNeurs,
                const std::valarray<T> & dNeurs,
                const std::valarray<T> & wConns,
                const T &  dt,
                const T & st,
                std::valarray<T> & VNeurs,
                std::valarray<T> & UNeurs_,
                std::valarray<bool> & mNeurs,
                std::valarray<T> & INeurs,
                T & t,
                std::deque<std::pair<T,std::valarray<T>>> & oscillograms
            ) = 0 ;

            /*! \~russian
             * \brief Интерфейс реализации модели UNN270117 методом Рунге-Кутты 4 5.
             *
             * \param nNeurs количество нейронов.
             * \param nNeursExc количество возбуждающих нейронов.
             * \param Nastr количество астроцитов
             * \param VNeursPeak предельное значение потенциала на нейронах.
             * \param VNeursReset значение потенциала на нейронах после спайка.
             * \param dt шаг по времени.
             * \param st время, до которого будет выполняться данная функция
             * (конечное модельное время для данного "шага").
             * \param[out] t текущее модельное время.
             * \param Cm Удельная мембранная емкость.
             * \param g_Na Проводимость ионного натриевого тока.
             * \param g_K Проводимость ионного калиевого тока.
             * \param g_leak Проводимость ионного тока утечки.
             * \param Iapp Постоянный внешний ток.
             * \param E_Na Реверсивный потенциал натрия.
             * \param E_K Реверсивный потенциал калия.
             * \param E_L Реверсивный потенциал утечки.
             * \param Esyn Реверсивный синаптический потенциал для тормозной связи.
             * \param theta_syn Сдвиг в формуле для синаптического тока.
             * \param k_syn Коэффициент в экспоненте в формуле синаптического тока.
             * \param alphaGlu Параметр в формуле для потока (I_glu) внешнего вещества (например, глутамата).
             * \param alphaG Параметр в формуле для внеклеточной концентрации
             * нейропередатчика в окрестности нейрона.
             * \param bettaG Параметр в формуле для внеклеточной концентрации
             * нейропередатчика в окрестности нейрона.
             * \param tauIP3 Параметр в дифференциальном уравнении на концентрацию
             * инозитол 1,4,5-трифосфата .
             * \param IP3ast Параметр в дифференциальном уравнении на концентрацию
             * инозитол 1,4,5-трифосфата . Равновесное значение ИТФ.
             * \param a2 Параметр в дифференциальном уравнении на долю IP3 в
             * эндоплазматическом ретикулуме, которая не была инактивирована кальцием Ca2+ .
             * \param d1 Параметр в дифференциальном уравнении на долю IP3 в
             * эндоплазматическом ретикулуме, которая не была инактивирована кальцием Ca2+ .
             * \param d2 Параметр в дифференциальном уравнении на долю IP3 в
             * эндоплазматическом ретикулуме, которая не была инактивирована кальцием Ca2+ .
             * \param d3 Параметр в дифференциальном уравнении на долю IP3 в
             * эндоплазматическом ретикулуме, которая не была инактивирована кальцием Ca2+ .
             * \param d5 Параметр в формуле для потока (I_channel) кальция из
             * эндоплазматического ретикулума в цитоплазму.
             * \param dCa Параметр в формуле для потока (I_Cadif) разности
             * концентрации кальция между астроцитами.
             * \param dIP3 Параметр в формуле для потока (I_IP3dif) разности
             * концентрации иноситола 1,4,5-трифосфата.
             * \param c0 Параметр в формуле для потока (I_channel) кальция из
             * эндоплазматического ретикулума в цитоплазму, а также в формуле
             * для пассивного тока (I_leak), соответствующего градиентному
             * переносу кальция через нейтральные каналы мембраны ЭР.
             * \param c1 Параметр в формуле для потока (I_channel) кальция из
             * эндоплазматического ретикулума в цитоплазму, а также в формуле
             * для пассивного тока (I_leak), соответствующего градиентному
             * переносу кальция через нейтральные каналы мембраны ЭР.
             * \param v1 Параметр в формуле для потока (I_channel) кальция из
             * эндоплазматического ретикулума в цитоплазму.
             * \param v4 Параметр в формуле для потока (I_PLC), зависящего от
             * концентрации кальция, в концентрацию ИТФ.
             * \param alpha Параметр в формуле для потока (I_PLC), зависящего
             * от концентрации кальция, в концентрацию ИТФ.
             * \param k4 Параметр в формуле для потока (I_PLC), зависящего от
             * концентрации кальция, в концентрацию ИТФ.
             * \param v2 Параметр в формуле для пассивного потока (I_leak),
             * соответствующего градиентному переносу кальция через нейтральные каналы мембраны ЭР.
             * \param v3 Параметр в формуле для обратного потока (I_pump)
             * кальция из цитоплазмы в ЭР.
             * \param k3 Параметр в формуле для обратного потока (I_pump)
             * кальция из цитоплазмы в ЭР.
             * \param v5 Параметр в формуле для пассивного потока (I_in)
             * процессов обмена кальция с внешней средой.
             * \param v6 Параметр в формуле для пассивного потока (I_in)
             * процессов обмена кальция с внешней средой.
             * \param k2 Параметр в формуле для пассивного потока (I_in)
             * процессов обмена кальция с внешней средой.
             * \param k1 Параметр в формуле для пассивного потока (I_out)
             * процессов обмена кальция с внешней средой.
             * \param IstimAmplitude Амплитуда тока, возникающего за счет
             * внешнего воздействия по распределению Пуассона.
             * \param IstimFrequency Частота возникновения внешнего тока на нейрон.
             * \param IstimDuration Длительность воздействия внешнего тока
             * на нейрон в миллисекундах.
             * \param[out] nextTimeEvent Массив с временем возникновения следующего
             * события (генерации импульса внешнего тока).
             * \param[out] Istim Текущее значение внешнего воздействия тока на нейроны.
             * \param wConns Матрица весов синаптических связей.
             * \param wAstrNeurs Матрица связи астроцитов с нейронами.
             * Коэффициент регуляции синаптической связи за счет воздействия астроцитов.
             * \param astrConns Матрица связи астроцитов между собой в астроцитарной сети.
             * \param[out] VNeurs Вектор мембранных потенциалов.
             * \param[out] mNeurs Маска, хранящая спайки.
             * \param[out] INeurs Вектор токов для нейров.
             * \param[out] m Активационная натриевая переменная (воротная переменная).
             * \param[out] h Инактивационная натриевая переменная (воротная переменная).
             * \param[out] n Активационная калиевая переменная (воротная переменная).
             * \param[out] G Внеклеточная концентрация нейропередатчика в окрестности нейрона.
             * \param[out] Ca Концентрация свободного цитозолического кальция
             * (free cytosolic calcium concentration) на астроците.
             * \param[out] IP3 Концентрация иноситол 1,4,5-трифосфата
             * (inositol 1,4,5-triphosphate concentration) на астроците.
             * \param[out] z Доля каналов на мембране эндоплазматического ретикулума,
             * находящихся в открытом состоянии (denotes the fraction of IP3
             * receptors in endoplasmic reticulum (ER) that have not been inactivated by Ca2+ ).
             * \param[out] oscillograms Вектор для хранения осцилограмм с напряжением на нейронах.
             */
            virtual void solveUNN270117
			(
            	const size_t& nNeurs,
				const size_t& nNeursExc,
				const size_t& Nastr,
				const T& VNeursPeak,
				const T& VNeursReset,
				const T& dt,
				const T& st,
				T& t,

				const T& Cm,
				const T& g_Na,
				const T& g_K,
				const T& g_leak,
				const T& Iapp,
				const T& E_Na,
				const T& E_K,
				const T& E_L,
				const T& Esyn,
				const T& theta_syn,
				const T& k_syn,

				const T& alphaGlu,
				const T& alphaG,
				const T& bettaG,

				const T& tauIP3,
				const T& IP3ast,
				const T& a2,
				const T& d1,
				const T& d2,
				const T& d3,
				const T& d5,

				const T& dCa,
				const T& dIP3,
				const T& c0,
				const T& c1,
				const T& v1,
				const T& v4,
				const T& alpha,
				const T& k4,
				const T& v2,
				const T& v3,
				const T& k3,
				const T& v5,
				const T& v6,
				const T& k2,
				const T& k1,

				const T& IstimAmplitude,
				const T& IstimFrequency,
				const T& IstimDuration,
				std::valarray<T>& nextTimeEvent,
				std::valarray<T>& Istim,

				const std::valarray<T>& wConns,
				const std::valarray<T>& wAstrNeurs,
				const std::valarray<bool>& astrConns,

				std::valarray<T>& VNeurs,
				std::valarray<bool>& mNeurs,
				std::valarray<T>& INeurs,
				std::valarray<T>& m,
				std::valarray<T>& h,
				std::valarray<T>& n,
				std::valarray<T>& G,

				std::valarray<T>& Ca,
				std::valarray<T>& IP3,
				std::valarray<T>& z,

				std::deque<std::pair<T,std::valarray<T>>>& oscillograms
            ) = 0 ;

            //! \~russian \brief Деструктор.
            virtual ~SolverImpl() {}

    };
} 
#endif

/*@}*/

