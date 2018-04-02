#define BOOST_TEST_MODULE NeursTest
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <string>
#include <valarray>
#include <fstream>
#include <iostream>

#include "../lib/impl/cpu/solvercpu.hpp"
#include "../lib/impl/cpu/solvercpu.cpp"
#include "../lib/solversbuffer.hpp"

#ifndef DEBUG
	#define DEBUG 0
#else
	#undef DEBUG
	#define DEBUG 0
#endif

/*! \~russian \brief Структура с параметрами для проведения тестирования функций модели UNN270117.
 *
 */
struct FromFileUNN270117{
	float dt;

	size_t nNeurs;

	float VReset;

	size_t Nastr;

	float Cm;
	float g_Na;
	float g_K;
	float g_leak;
	float Iapp;
	float E_Na;
	float E_K;
	float E_leak;
	float Esyn;
	float theta_syn;
	float k_syn;
	float alphaGlu;
	float alphaG;
	float bettaG;

	float tauIP3;
	float IP3ast;
	float a2;
	float d1;
	float d2;
	float d3;
	float d5;
	float dCa;
	float dIP3;
	float c0;
	float c1;
	float v1;
	float v4;
	float alpha;
	float k4;
	float v2;
	float v3;
	float k3;
	float v5;
	float v6;
	float k2;
	float k1;

	float Istim;

	std::valarray<float> wAstrNeurs;
	std::valarray<bool>  astrConns;
	std::valarray<float> wConns;

	//variables

	std::valarray<float> VNeurs;
	std::valarray<float> INeurs;
	std::valarray<float> m;
	std::valarray<float> h;
	std::valarray<float> n;
	std::valarray<float> G;
	std::valarray<float> Ca;
	std::valarray<float> IP3;
	std::valarray<float> z;

	void readFromFile(const std::string& filename){
		std::ifstream testFin;
		testFin.open(filename.c_str(), std::ios::in);
		if (!testFin.is_open()){
			std::cerr<<"ERROR in "<<__FUNCTION__<<"(const std::string&) : line "<<__LINE__<<", file "<<__FILE__<<std::endl;
			std::cerr<<"\tCan\'t open file with name"<<std::endl;
			std::cerr<<filename<<std::endl;
			throw;
		}
		testFin.close();

		SolversBuffer fin;
		fin.readFile(filename);

		fin >> dt;
		fin >> nNeurs;
		fin >> VReset;

		fin >> Nastr;
		fin >> Cm;
		fin >> g_Na;
		fin >> g_K;
		fin >> g_leak;
		fin >> Iapp;
		fin >> E_Na;
		fin >> E_K;
		fin >> E_leak;
		fin >> Esyn;
		fin >> theta_syn;
		fin >> k_syn;
		fin >> alphaGlu;
		fin >> alphaG;
		fin >> bettaG;

		fin >> tauIP3;
		fin >> IP3ast;
		fin >> a2;
		fin >> d1;
		fin >> d2;
		fin >> d3;
		fin >> d5;
		fin >> dCa;
		fin >> dIP3;
		fin >> c0;
		fin >> c1;
		fin >> v1;
		fin >> v4;
		fin >> alpha;
		fin >> k4;
		fin >> v2;
		fin >> v3;
		fin >> k3;
		fin >> v5;
		fin >> v6;
		fin >> k2;
		fin >> k1;

		fin >> Istim;

		wAstrNeurs.resize( nNeurs * Nastr );
		fin >> wAstrNeurs[0];

		astrConns.resize( Nastr * Nastr );
		fin >> astrConns[0];

		wConns.resize( nNeurs * nNeurs );
		fin >> wConns[0];

		VNeurs.resize( nNeurs );
		INeurs.resize( nNeurs );
		m.resize( nNeurs );
		h.resize( nNeurs );
		n.resize( nNeurs );
		G.resize( nNeurs );
		Ca.resize( Nastr );
		IP3.resize( Nastr );
		z.resize( Nastr );
	}
};

BOOST_AUTO_TEST_SUITE (SolverTest) 

BOOST_AUTO_TEST_CASE (TestSolverPCNNI2003E)
{
	std::cout<<"\n\033[32;1m TEST FOR UNN270117 FUNCTIONS: \033[0m\n";

    #if DEBUG 
        std::cout << "\n\033[32;1mFor testing uncomment line  '#define NN_TEST_SOLVERS' in the setting.h and rebuild libs \033[0m\n"; 
    #endif

        float x=1, y=2.1;
    BOOST_CHECK_CLOSE_FRACTION( x, 1, 1e-5 );

    // SETTING ALL PARAMETERS
    float dt = 1.0;

    size_t nNeurs = 1;

    float VReset = -65.0;

    size_t Nastr = 1;

    float Cm 		= 	1.0;
    float g_Na 		= 	120.0;
    float g_K 		= 	36.0;
    float g_leak 	= 	0.3;
    float Iapp 		= 	2.0;
    float E_Na		=	55.0;
    float E_K		=	-77.0;
    float E_leak	=	-54.4;
    float Esyn		=	-90.0;
    float theta_syn =	0.0;
    float k_syn		=	0.2;
    float alphaGlu	=	0.01;
    float alphaG	=	0.025;
    float bettaG	=	0.5;

    float tauIP3	=	7142.85714;
    float IP3ast	=	0.16;
    float a2		=	0.00014;
    float d1		=	0.13;
    float d2		=	1.049;
    float d3		=	0.9434;
    float d5		=	0.082;
    float dCa		=	0.000001;
    float dIP3		=	0.00012;
    float c0		=	2.0;
    float c1		=	0.185;
    float v1		=	0.006;
    float v4		=	0.0003;
    float alpha		=	0.8;
    float k4		=	0.0011;
    float v2		=	0.00011;
    float v3		=	0.0022;
    float k3		=	0.1;
    float v5		=	0.000025;
    float v6		=	0.0002;
    float k2		=	1.0;
    float k1		=	0.0005;

    float Istim 	=	3.0;

    std::valarray<float> wAstrNeurs;
    wAstrNeurs.resize( Nastr * nNeurs );
    wAstrNeurs[0] = 3.0;

    std::valarray<bool> astrConns;
    astrConns.resize(Nastr*Nastr);
    astrConns[0] = 0;

    std::valarray<float> wConns;
    wConns.resize(nNeurs * nNeurs);
    wConns[0] = 0.05;

    // variables

    std::valarray<float> VNeurs;
    VNeurs.resize(nNeurs);

    std::valarray<float> INeurs;
    INeurs.resize(nNeurs);

    std::valarray<float> m;
    m.resize(nNeurs);

    std::valarray<float> h;
    h.resize(nNeurs);

    std::valarray<float> n;
    n.resize(nNeurs);

    std::valarray<float> G;
    G.resize(nNeurs);

    std::valarray<float> Ca;
    Ca.resize(Nastr);

    std::valarray<float> IP3;
    IP3.resize(Nastr);

    std::valarray<float> z;
    z.resize(Nastr);

    // USING INFORMATION FROM FILE
    FromFileUNN270117 FF;
    std::string infile = "./solverUNN270117RK.in";
    FF.readFromFile(infile);

    // TESTS
    float outValFF;
    float outVal;
    std::string outDataFile = "./solverUNN270117RK.end";
    SolversBuffer fin;
    fin.readFile(outDataFile);

    // 1 IN
    IP3.resize( 2*Nastr );
    Ca.resize( 2*Nastr );
    z.resize( 2*Nastr );
    fin >> IP3[0] >> IP3[1];
    fin >> Ca[0] >> Ca[1];
    fin >> z[0] >> z[1];
    // 1 OUT     J_channel

    using std::cout;
    using std::endl;

	#if DEBUG == 1
    cout<<"\033[33;1m";
    std::cout<<"c1 = "<<c1<<std::endl;
    cout<<"v1 = "<<v1<<endl;
    cout<<"IP3 = "<<IP3[0]<<", "<<IP3[1]<<endl;
    cout<<"Ca = "<<Ca[0]<<", "<<Ca[1]<<endl;
    cout<<"z = "<<z[0]<<", "<<z[1]<<endl;
    cout<<"c0 = "<<c0<<endl;
    cout<<"d1 = "<<d1<<endl;
    cout<<"d5 = "<<d5<<endl;
    cout<<"\033[0m";
	#endif
    for (size_t i=0; i < 2; ++i){
    	fin >> outValFF;
		outVal = NNSimulator::NARK45::J_channel<float>(c1, v1, IP3[i], Ca[i], z[i], c0, d1, d5);
		BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 1 (J_channel) : исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 2 OUT     J_PLC
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"v4 = "<<v4<<endl;
    cout<<"Ca = "<<Ca[0]<<", "<<Ca[1]<<endl;
    cout<<"alpha = "<<alpha<<endl;
    cout<<"k4 = "<<k4<<endl;
    cout<<"\033[0m";
	#endif
    for(size_t i=0; i < 2; ++i){
    	fin >> outValFF;
    	outVal = NNSimulator::NARK45::J_PLC<float>(v4, Ca[i], alpha, k4);
    	BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 2 (J_PLC): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 3 OUT J_leak
	#if DEBUG == 1
    cout<<"\033[33;1m";
    std::cout<<"c1 = "<<c1<<std::endl;
    std::cout<<"v2 = "<<v2<<std::endl;
    std::cout<<"c0 = "<<c0<<std::endl;
    cout<<"Ca = "<<Ca[0]<<", "<<Ca[1]<<endl;
    cout<<"\033[0m";
	#endif
    for(size_t i=0; i < 2; ++i){
    	fin >> outValFF;
    	outVal = NNSimulator::NARK45::J_leak(c1, v2, c0, Ca[i]);
    	BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 3 (J_leak): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 4 OUT J_pump
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"v3 = "<<v3<<endl;
    cout<<"Ca = "<<Ca[0]<<", "<<Ca[1]<<endl;
    cout<<"k3 = "<<k3<<endl;
    cout<<"\033[0m";
	#endif
    for(size_t i=0; i < 2; ++i){
    	fin >>outValFF;
    	outVal = NNSimulator::NARK45::J_pump(v3, Ca[i], k3);
    	BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 4 (J_pump): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 5 OUT J_in
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"v5 = "<<v5<<endl;
    cout<<"v6 = "<<v6<<endl;
    cout<<"IP3 = "<<IP3[0]<<", "<<IP3[1]<<endl;
    cout<<"k2 = "<<k2<<endl;
    cout<<"\033[0m";
	#endif
    for(size_t i=0; i < 2; ++i){
    	fin >> outValFF;
    	outVal = NNSimulator::NARK45::J_in(v5, v6, IP3[i], k2);
    	BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 5 (J_in): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 6 OUT J_out
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"k1 = "<<k1<<endl;
    cout<<"Ca = "<<Ca[0]<<", "<<Ca[1]<<endl;
    cout<<"\033[0m";
	#endif
    for(size_t i=0; i < 2; ++i){
    	fin >> outValFF;
    	outVal = NNSimulator::NARK45::J_out(k1, Ca[i]);
    	BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 6 (J_out): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 7 IN
    G.resize(2 * nNeurs);
    fin >> G[0] >> G[1];
    // 7 OUT J_Glu
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"ind = 0"<<endl;
    cout<<"alphaGlu = "<<alphaGlu<<endl;
    cout<<"G = "<<G[0]<<", "<<G[1]<<endl;
    cout<<"wAstrNeurs = "<<wAstrNeurs[0]<<endl;
    cout<<"nNeurs = "<<nNeurs<<endl;
    cout<<"Nastr = "<<Nastr<<endl;
    cout<<"\033[0m";
	#endif
    for(size_t i=0; i < 2; ++i){
    	std::valarray<float> Gtemp;
    	Gtemp.resize(1);
    	Gtemp[0] = G[i];

    	fin >> outValFF;
    	outVal = NNSimulator::NARK45::J_Glu<float>(0, alphaGlu, Gtemp, wAstrNeurs, nNeurs, Nastr);
    	BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 7 (J_Glu): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 8 OUT J_Cadif
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"ind = 0"<<endl;
    cout<<"dCa = "<<dCa<<endl;
    cout<<"Ca = "<<Ca[0]<<", "<<Ca[1]<<endl;
    cout<<"astrConns = "<<astrConns[0]<<endl;
    cout<<"Nastr = "<<Nastr<<endl;
    cout<<"\033[0m";
	#endif
    for(size_t i=0; i < 2; ++i){
    	std::valarray<float> CaTemp;
    	CaTemp.resize(1);
    	CaTemp[0] = Ca[i];

    	fin >> outValFF;
    	outVal = NNSimulator::NARK45::J_Cadif(0, dCa, CaTemp, astrConns, Nastr);
    	BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 8 (J_Cadif): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 9 OUT J_IP3dif
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"ind = 0"<<endl;
    cout<<"dIP3 = "<<dIP3<<endl;
    cout<<"IP3 = "<<IP3[0]<<", "<<IP3[1]<<endl;
    cout<<"astrConns = "<<astrConns[0]<<endl;
    cout<<"Nastr = "<<Nastr<<endl;
    cout<<"\033[0m";
	#endif
    for(size_t i=0; i < 2; ++i){
    	std::valarray<float> IP3Temp;
    	IP3Temp.resize(1);
    	IP3Temp[0] = IP3[i];

    	fin >> outValFF;
    	outVal = NNSimulator::NARK45::J_IP3dif(0, dIP3, IP3Temp, astrConns, Nastr);
    	BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 9 (J_IP3dif): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 10 OUT F_Ca
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"ind = 0"<<endl;
    cout<<"Ca = "<<Ca[0]<<", "<<Ca[1]<<endl;
    cout<<"IP3 = "<<IP3[0]<<", "<<IP3[1]<<endl;
    cout<<"z = "<<z[0]<<", "<<z[1]<<endl;
    cout<<"c1 = "<<c1<<endl;
    cout<<"v1 = "<<v1<<endl;
    cout<<"c0 = "<<c0<<endl;
    cout<<"d1 = "<<d1<<endl;
    cout<<"d5 = "<<d5<<endl;
    cout<<"v3 = "<<v3<<endl;
    cout<<"k3 = "<<k3<<endl;
    cout<<"v2 = "<<v2<<endl;
    cout<<"v5 = "<<v5<<endl;
    cout<<"v6 = "<<v6<<endl;
    cout<<"k2 = "<<k2<<endl;
    cout<<"k1 = "<<k1<<endl;
    cout<<"dCa = "<<dCa<<endl;
    cout<<"astrConns = "<<astrConns[0]<<endl;
    cout<<"Nastr = "<<Nastr<<endl;
    cout<<"\033[0m";
	#endif
    for(size_t i=0; i < 2; ++i){
    	std::valarray<float> CaTemp;
    	CaTemp.resize(1);
    	CaTemp[0] = Ca[i];

    	fin >> outValFF;
    	outVal = NNSimulator::NARK45::F_Ca(0, CaTemp, IP3[i], z[i],
    			c1, v1, c0, d1, d5, v3, k3, v2, v5, v6, k2, k1,
				dCa, astrConns, Nastr);
    	BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 10 (F_Ca): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 11 OUT F_IP3
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"ind = 0"<<endl;
    cout<<"Ca = "<<Ca[0]<<", "<<Ca[1]<<endl;
    cout<<"IP3 = "<<IP3[0]<<", "<<IP3[1]<<endl;
    cout<<"G = "<<G[0]<<", "<<G[1]<<endl;
    cout<<"v4 = "<<v4<<endl;
    cout<<"alpha = "<<alpha<<endl;
    cout<<"k4 = "<<k4<<endl;
    cout<<"dIP3 = "<<dIP3<<endl;
    cout<<"alphaGlu = "<<alphaGlu<<endl;
    cout<<"IP3ast = "<<IP3ast<<endl;
    cout<<"tauIP3 = "<<tauIP3<<endl;
    cout<<"astrConns = "<<astrConns[0]<<endl;
    cout<<"wAstrNeurs = "<<wAstrNeurs[0]<<endl;
    cout<<"Nastr = "<<Nastr<<endl;
    cout<<"nNeurs = "<<nNeurs<<endl;
    cout<<"\033[0m";
	#endif
    for(size_t i=0; i < 2; ++i){
    	std::valarray<float> IP3Temp;
    	IP3Temp.resize(1);
    	IP3Temp[0] = IP3[i];
    	std::valarray<float> GTemp;
    	GTemp.resize(1);
    	GTemp[0] = G[i];

    	fin >> outValFF;
    	outVal = NNSimulator::NARK45::F_IP3(0, Ca[i], IP3Temp, GTemp,
    			v4, alpha, k4, dIP3, alphaGlu,
				IP3ast, tauIP3, astrConns, wAstrNeurs, Nastr, nNeurs);
    	BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 11 (F_IP3): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 12 OUT F_z
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"Ca = "<<Ca[0]<<", "<<Ca[1]<<endl;
    cout<<"IP3 = "<<IP3[0]<<", "<<IP3[1]<<endl;
    cout<<"z = "<<z[0]<<", "<<z[1]<<endl;
    cout<<"a2 = "<<a2<<endl;
    cout<<"d2 = "<<d2<<endl;
    cout<<"d1 = "<<d1<<endl;
    cout<<"d3 = "<<d3<<endl;
    cout<<"\033[0m";
	#endif
    for(size_t i=0; i < 2; ++i){
    	fin >> outValFF;
    	outVal = NNSimulator::NARK45::F_z(Ca[i], IP3[i], z[i], a2, d2, d1, d3);
    	BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 12 (F_z): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    //IN Vmbr
    VNeurs.resize( 2*nNeurs );
    fin >> VNeurs[0] >> VNeurs[1];
    // 13 OUT H(V)
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"Vmbr = "<<VNeurs[0]<<", "<<VNeurs[1]<<endl;
    cout<<"\033[0m";
	#endif
    for(size_t i=0; i < 2; ++i){
    	fin >> outValFF;
    	outVal = NNSimulator::NARK45::Hpot(VNeurs[i]);
    	BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 13 (Hpot): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 14 OUT F_G
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"alphaG = "<<alphaG<<endl;
    cout<<"G = "<<G[0]<<", "<<G[1]<<endl;
    cout<<"bettaG = "<<bettaG<<endl;
    cout<<"Vmbr = "<<VNeurs[0]<<", "<<VNeurs[1]<<endl;
    cout<<"\033[0m";
	#endif
    for(size_t i=0; i < 2; ++i){
    	fin >> outValFF;
    	outVal = NNSimulator::NARK45::F_G(alphaG, G[i], bettaG, VNeurs[i]);
    	BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 14 (F_G): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 15 IN
    nNeurs = 2;
    VNeurs.resize( nNeurs );
    wConns.resize( nNeurs*nNeurs );
    std::vector<float> tmp(2);
    tmp[0] = Ca[0];
    tmp[1] = Ca[1];
    Ca.resize( nNeurs + 1 );
    Ca[0] = tmp[0];
    Ca[1] = tmp[1];
    Ca[2] = 0.0;
    std::valarray<float> VmbrK(3), VmbrL(3), wConnsTemp(3);
    fin >> VmbrK[0] >> VmbrK[1] >> VmbrK[2];
    fin >> VmbrL[0] >> VmbrL[1] >> VmbrL[2];
    fin >> wConnsTemp[0] >> wConnsTemp[1] >> wConnsTemp[2];
    size_t ind = 0;

    tmp[0] = wAstrNeurs[0];
    wAstrNeurs.resize( nNeurs * Nastr );
    wAstrNeurs[0] = tmp[0];
    wAstrNeurs[1] = wAstrNeurs[0];

    // 15 OUT I_syn
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"ind = "<<ind<<endl;
    cout<<"Ca = "<<Ca[0]<<", "<<Ca[1]<<", "<<Ca[2]<<endl;
    cout<<"Vmbr K = "<<VmbrK[0]<<", "<<VmbrK[1]<<", "<<VmbrK[2]<<endl;
    cout<<"Vmbr L = "<<VmbrL[0]<<", "<<VmbrL[1]<<", "<<VmbrL[2]<<endl;
    cout<<"wConns = "<<wConnsTemp[0]<<", "<<wConnsTemp[1]<<", "<<wConnsTemp[2]<<endl;
    cout<<"wAstrNeurs = "<<wAstrNeurs[0]<<", "<<wAstrNeurs[1]<<endl;
    cout<<"Esyn = "<<Esyn<<endl;
    cout<<"theta_syn = "<<theta_syn<<endl;
    cout<<"k_syn = "<<k_syn<<endl;
    cout<<"Nastr = "<<Nastr<<endl;
    cout<<"nNeurs = "<<nNeurs<<endl;
    cout<<"\033[0m";
	#endif
    for(size_t i=0; i < 3; ++i){
    	VNeurs[0] = VmbrK[i];
    	VNeurs[1] = VmbrL[i];

    	wConns[0] = 0.0;
    	wConns[1] = wConnsTemp[i];
    	wConns[2] = wConnsTemp[i];
    	wConns[3] = 0.0;

    	std::valarray<float> CaTemp(1);
    	CaTemp[0] = Ca[i];

    	fin >> outValFF;
    	outVal = NNSimulator::NARK45::I_neurs(ind, CaTemp, VNeurs, wConns, wAstrNeurs,
    			Esyn, theta_syn, k_syn, Nastr, nNeurs);
    	BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    }
    std::cout << "\033[31;1m Тест 15 (I_syn): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    //IN воротные переменные
    nNeurs = 1;
    m.resize( nNeurs );
    h.resize( nNeurs );
    n.resize( nNeurs );
    VNeurs.resize( nNeurs );
    fin >> m[0] >> h[0] >> n[0];
    fin >> VNeurs[0];
    // 16 OUT alphaM
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"Vmbr = "<<VNeurs[0]<<endl;
    cout<<"\033[0m";
	#endif
    fin >> outValFF;
    outVal = NNSimulator::NARK45::alphaM(VNeurs[0]);
    BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    std::cout << "\033[31;1m Тест 16 (alphaM): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 17 OUT bettaM
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"Vmbr = "<<VNeurs[0]<<endl;
    cout<<"VReset = "<<VReset<<endl;
    cout<<"\033[0m";
	#endif
    fin >> outValFF;
    outVal = NNSimulator::NARK45::bettaM(VNeurs[0], VReset);
    BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    std::cout << "\033[31;1m Тест 17 (bettaM): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 18 OUT F_m
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"Vmbr = "<<VNeurs[0]<<endl;
    cout<<"m = "<<m[0]<<endl;
    cout<<"VReset = "<<VReset<<endl;
    cout<<"\033[0m";
	#endif
    fin >> outValFF;
    outVal = NNSimulator::NARK45::F_m(VNeurs[0], m[0], VReset);
    BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    std::cout << "\033[31;1m Тест 18 (F_m): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 19 OUT alphaH
	#if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"Vmbr = "<<VNeurs[0]<<endl;
    cout<<"VReset = "<<VReset<<endl;
    cout<<"\033[0m";
	#endif
    fin >> outValFF;
    outVal = NNSimulator::NARK45::alphaH(VNeurs[0], VReset);
    BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    std::cout << "\033[31;1m Тест 19 (alphaH): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 20 OUT bettaH
	#if DEBUG == 1
	cout<<"\033[33;1m";
	cout<<"Vmbr = "<<VNeurs[0]<<endl;
	cout<<"\033[0m";
	#endif
    fin >> outValFF;
    outVal = NNSimulator::NARK45::bettaH(VNeurs[0]);
    BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    std::cout << "\033[31;1m Тест 20 (bettaH): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 21 OUT F_h
	#if DEBUG == 1
	cout<<"\033[33;1m";
	cout<<"Vmbr = "<<VNeurs[0]<<endl;
	cout<<"h = "<<h[0]<<endl;
	cout<<"VReset = "<<VReset<<endl;
	cout<<"\033[0m";
	#endif
    fin >> outValFF;
    outVal = NNSimulator::NARK45::F_h(VNeurs[0], h[0], VReset);
    BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    std::cout << "\033[31;1m Тест 21 (F_h): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 22 OUT alphaN
	#if DEBUG == 1
	cout<<"\033[33;1m";
	cout<<"Vmbr = "<<VNeurs[0]<<endl;
	cout<<"\033[0m";
	#endif
    fin >> outValFF;
    outVal = NNSimulator::NARK45::alphaN(VNeurs[0]);
    BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    std::cout << "\033[31;1m Тест 22 (alphaN): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 23 OUT bettaN
	#if DEBUG == 1
	cout<<"\033[33;1m";
	cout<<"Vmbr = "<<VNeurs[0]<<endl;
	cout<<"VReset = "<<VReset<<endl;
	cout<<"\033[0m";
	#endif
    fin >> outValFF;
    outVal = NNSimulator::NARK45::bettaN(VNeurs[0], VReset);
    BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    std::cout << "\033[31;1m Тест 23 (bettaN): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 24 OUT F_n
	#if DEBUG == 1
	cout<<"\033[33;1m";
	cout<<"Vmbr = "<<VNeurs[0]<<endl;
	cout<<"n = "<<n[0]<<endl;
	cout<<"VReset = "<<VReset<<endl;
	cout<<"\033[0m";
	#endif
	fin >> outValFF;
    outVal = NNSimulator::NARK45::F_n(VNeurs[0], n[0], VReset);
    BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    std::cout << "\033[31;1m Тест 24 (F_n): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    // 25 OUT F_mbr
    nNeurs = 2;
    VNeurs.resize( nNeurs );
    VNeurs[0] = -70;
    VNeurs[1] = 10;
    Ca[0] = 0.3;
    wConns[0] = 0;
    wConns[1] = 0.05;
    wConns[2] = 0.05;
    wConns[3] = 0;
    #if DEBUG == 1
    cout<<"\033[33;1m";
    cout<<"ind = 0"<<endl;
    cout<<"Vmbr = "<<VNeurs[0]<<endl;
    cout<<"m = "<<m[0]<<endl;
    cout<<"h = "<<h[0]<<endl;
    cout<<"n = "<<n[0]<<endl;
    cout<<"Istim = "<<Istim<<endl;
    cout<<"Ca = "<<Ca[0]<<endl<<endl;
    cout<<"Cm = "<<Cm<<endl;
    cout<<"g_Na = "<<g_Na<<endl;
    cout<<"E_Na = "<<E_Na<<endl;
    cout<<"g_K = "<<g_K<<endl;
    cout<<"E_K = "<<E_K<<endl;
    cout<<"g_leak = "<<g_leak<<endl;
    cout<<"E_leak = "<<E_leak<<endl;
    cout<<"Iapp = "<<Iapp<<endl;
    cout<<"wConns = "<<wConns[0]<<endl;
    cout<<"wAstrNeurs = "<<wAstrNeurs[0]<<endl;
    cout<<"Esyn = "<<Esyn<<endl;
    cout<<"theta_syn = "<<theta_syn<<endl;
    cout<<"k_syn = "<<k_syn<<endl;
    cout<<"Nastr = "<<Nastr<<endl;
    cout<<"nNeurs = "<<nNeurs<<endl;
    cout<<"\033[0m";
	#endif
    fin >> outValFF;
    outVal = NNSimulator::NARK45::F_mbr(0, VNeurs, m[0], h[0], n[0], Istim, Ca,
    		Cm, g_Na, E_Na, g_K, E_K, g_leak, E_leak, Iapp,
			wConns, wAstrNeurs, Esyn, theta_syn, k_syn,
			Nastr, nNeurs);
    BOOST_CHECK_CLOSE_FRACTION( outVal , outValFF, 1e-5 );
    std::cout << "\033[31;1m Тест 25 (F_mbr): исполнился \033[0m ";
#if DEBUG >= 1
    std::cout << std::endl << std::endl;
#else
    std::cout << std::flush;
#endif

    BOOST_AUTO_TEST_SUITE_END();
}
