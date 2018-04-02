#define BOOST_TEST_MODULE NeursTest
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <iostream>
#include <sstream>
#include <vector>
#include <type_traits>
#include "../lib/solver.hpp"
#include "../lib/formatstream.hpp"
#include "../lib/solversbuffer.hpp"

#ifndef DEBUG
	#define DEBUG 0
#else
	#undef DEBUG
	#define DEBUG 0
#endif

BOOST_AUTO_TEST_SUITE (SolverTest) 



BOOST_AUTO_TEST_CASE (TestSolverPCNNI2003E)
{
	using ValType = double;
	const double difference = 1e-5;

    #if DEBUG 
        std::cout << "\n\033[32;1mFor testing uncomment line  '#define NN_TEST_SOLVERS' in the setting.h and rebuild libs \033[0m\n";
    #endif

    std::string inFileName = "./realModel/realModel_UNN270117_1.sin";				// YOU CAN CHANGE IT
    SolversBuffer inBuff;
    inBuff.readFile(inFileName);

    #if DEBUG 
        std::cout << "\n\033[33;1m * * * SolverUNN270117 * * *\n\n\033[0m";
	#endif
        std::cout<< "\n\033[33;1m  WARNING: be sure that you launch SolverUNN270117 with #define DEBUG 1 in solvercpu.cpp file \n\n\033[0m";
	#if DEBUG
        std::cout << "In data:\n\033[36m" <<  inBuff.str() << "\033[0m" << std::endl;
    #endif

    size_t outNumberSolver;
    inBuff >> outNumberSolver;

    auto s = NNSimulator::Solver<ValType>::createItem(
            static_cast<typename NNSimulator::Solver<ValType>::ChildId>( outNumberSolver )
    ); 
    s->read( inBuff );

    std::ofstream fout;
    std::string outFilename = "test_UNN270117_outData.sout";						// YOU CAN CHANGE IT
    fout.open(outFilename, std::ios::out);
    if (!fout.is_open()){
    	std::cerr<<"ERROR in "<<__FUNCTION__<<" : line "<<__LINE__<<", file "<<__FILE__<<std::endl;
    	std::cerr<<"\tcan\'t create file with name"<<std::endl;
    	std::cerr<<outFilename<<std::endl;
    	std::cerr<<"\tMake sure that all needed directories exist and you have rights to create that file"<<std::endl<<std::endl;
    	throw;
    }
    s->solve(std::move(fout));

    // READING PARAMETERS AND COMPARING Vmbr

    NNSimulator::DataUNN270117<ValType> unnData;

    ValType VmbrFF, Vmbr;
    std::string VmbrFFData = "./realModel/boost_test_realModel_UNN270117.inVmbr";	// YOU CAN CHANGE IT
    std::ifstream finFF;
    finFF.open(VmbrFFData, std::ios::in);
    if (!finFF.is_open()){
    	std::cerr<<"ERROR in "<<__FUNCTION__<<" : line "<<__LINE__<<", file "<<__FILE__<<std::endl;
    	std::cerr<<"\tCan\'t open file with name"<<std::endl;
    	std::cerr<<VmbrFFData<<std::endl<<std::endl;
    	throw;
    }

    std::ifstream finModelData;
    finModelData.open(outFilename, std::ios::in);

    size_t counter = 0;
    size_t vst;
    finModelData >> vst >> vst;
    while( finModelData >> unnData ){
    	Vmbr = unnData.VNeurs[0];
    	finFF >> VmbrFF;

    	BOOST_CHECK_CLOSE_FRACTION( Vmbr, VmbrFF, difference );

		#if DEBUG
    		std::cout<<"\033[31;1m Step "<<counter<<" checking finished \033[0m"<<std::endl;
		#endif
    	++counter;
    	finModelData >> vst >> vst;
    }

    finModelData.close();
    finFF.close();

    BOOST_AUTO_TEST_SUITE_END();
}
