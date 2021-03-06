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

#define DEBUG 0

BOOST_AUTO_TEST_SUITE (SolverTest) 



BOOST_AUTO_TEST_CASE (TestSolverPCNNI2003E)
{
    #if DEBUG 
        std::cout << "\n\033[32;1mFor testing uncomment line  '#define NN_TEST_SOLVERS' in the setting.h and rebuild libs \033[0m\n";
    #endif

    std::string inFileName = "solverpcnni2003e.in";
    SolversBuffer inBuff;
    inBuff.readFile(inFileName);

    #if DEBUG 
        std::cout << "\n\033[33;1m * * * SolverPCNN * * *\n\n\033[0m";
        std::cout << "In data:\n\033[36m" <<  inBuff.str() << "\033[0m" << std::endl;
    #endif

    size_t outNumberSolver;
    inBuff >> outNumberSolver;

    auto s = NNSimulator::Solver<float>::createItem( 
            static_cast<typename NNSimulator::Solver<float>::ChildId>( outNumberSolver ) 
    ); 
    s->read( inBuff );
    s->solve();
    std::stringstream outBuff;
    s->write( outBuff );

    #if DEBUG 
        std::cout << "End data:\n\033[34m";
        std::cout << outBuff.str() << "\033[0m" << std::endl;
    #endif

    std::string pyOutFileName = "solverpcnni2003e.end";
    SolversBuffer pyOutBuff;
    pyOutBuff.readFile(pyOutFileName);

    #if DEBUG 
        std::cout << "End data from the testing program (Python):\n\033[34m";
        std::cout << pyOutBuff.str() << "\033[0m" << std::endl;
    #endif

    float outSolverBuffVal, outPyBuffVal;
    while( !pyOutBuff.eof() )
    {
        outBuff >> outSolverBuffVal;
        pyOutBuff >> outPyBuffVal;
        BOOST_CHECK_CLOSE_FRACTION( outSolverBuffVal, outPyBuffVal, 1e-5 );
    }

    BOOST_AUTO_TEST_SUITE_END();
}
