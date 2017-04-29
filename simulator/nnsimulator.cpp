#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <regex>
#include "../lib/solver.hpp"
#include "../lib/formatstream.hpp"
#include "../lib/solversbuffer.hpp"

#define DEBUG 0

int main(int argc, char* argv[])
{

    if( !argv[1] ) 
    {
        std::cout << "set file name\n";
        exit(1);
    }
    std::string inFileName(argv[1]);

    std::regex exts("\\.sin");
    std::string outFileName, spksFileName, oscsFileName;
    if( std::regex_search( inFileName, exts ) )
    {
        outFileName = std::regex_replace(
            inFileName, 
            exts, 
            ".sout"
        );
        spksFileName = std::regex_replace(
            inFileName, 
            exts, 
            ".spks"
        );
        oscsFileName = std::regex_replace(
            inFileName, 
            exts, 
            ".oscs"
        );
    }
    else 
    {
        std::cout << "Warning: the extension must be 'sout'.\n";
        std::cout << "outFileName = 'tmp.sout'.\n";
        outFileName = "tmp.sout";
    }

    SolversBuffer inBuff;
    inBuff.readFile(inFileName);
    #if DEBUG 
        std::cout << "\n\033[32;1mSolverPCNN = { ";
        std::cout << "\033[32;2m" <<  inBuff.str() << " \033[32;1m}\033[0m" << std::endl;
    #endif

    size_t outNumberSolver; //, outNumberData;
    inBuff >> outNumberSolver;

    auto s = NNSimulator::Solver<float>::createItem( 
            static_cast<typename NNSimulator::Solver<float>::ChildId>( outNumberSolver ) 
    ); 
    s->read( inBuff );

    std::ofstream outFileStream( outFileName );

    s->solve( std::move(outFileStream) );
    //s->solve( std::move(std::cout) );

    auto citsSpikes = s->getSpikes();
    std::ofstream spksFileStream(spksFileName);
    std::copy
    (
        citsSpikes.first,
        citsSpikes.second,
        std::ostream_iterator<std::pair<size_t,float>>( spksFileStream, "\n" )
    );
     
    auto citsOscillograms = s->getOscillograms();
    std::ofstream oscsFileStream( oscsFileName );
    std::copy
    (
        citsOscillograms.first,
        citsOscillograms.second,
        std::ostream_iterator<std::pair<float,std::valarray<float>>>( oscsFileStream, "\n" )
    );


    #if DEBUG 
        std::stringstream outBuff;
        s->write( outBuff );
        std::cout << "\033[34m";
        std::cout << outBuff.str() << "\033[0m" << std::endl;
    #endif


}
