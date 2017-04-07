#include <iostream>
#include <sstream>
#include <fstream>
#include <type_traits>
#include "../lib/solver.hpp"
#include "../lib/formatstream.hpp"
#include "../lib/solversbuffer.hpp"

#define DEBUG 1


int main(int argc, char* argv[])
{
    std::string inFileName = "pcnnmodel.in";
    SolversBuffer inBuff;
    inBuff.readFile(inFileName);
    #if DEBUG 
        std::cout << "\nSolverPCNN\n\033[31;1m";
        std::cout << "\033[31m" <<  inBuff.str() << "\033[0m" << std::endl;
    #endif

    size_t outNumberSolver;
    inBuff >> outNumberSolver;

    std::cout << "numberSolver = " << outNumberSolver << std::endl;

    auto s = NNSimulator::Solver<float>::createItem( 
            static_cast<typename NNSimulator::Solver<float>::ChildId>( outNumberSolver ) 
    ); 
    s->read( inBuff );

    std::string outFileName = "pcnnmodel.out";
    std::ofstream outFileStream( outFileName );

    s->solve( std::move(outFileStream) );
    //s->solve( std::move(std::cout) );

    std::stringstream outBuff;
    s->write( outBuff );

    #if DEBUG 
        std::cout << "\033[34m";
        std::cout << outBuff.str() << "\033[0m" << std::endl;
    #endif


}
