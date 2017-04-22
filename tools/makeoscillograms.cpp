#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <regex>
#include "../lib/data.hpp"
#include "../lib/formatstream.hpp"
#include "../lib/solversbuffer.hpp"

#define DEBUG 1


int main(int argc, char* argv[])
{
    if( !argv[1] ) 
    {
        std::cout << "error: Set file name\n";
        exit(1);
    }

    std::string inFileName = argv[1];
    std::regex exts("\\.sout");
    std::string outFileName;
    if( std::regex_search( inFileName, exts ) )
    {
        outFileName = std::regex_replace(
            inFileName, 
            exts, 
            ".oscs"
        );
    }
    else 
    {
        std::cout << "Warning: the extension must be 'sout'.\n";
        std::cout << "outFileName = 'tmp.oscs'.\n";
        outFileName = "tmp.oscs";
    }

    SolversBuffer inBuff;
    inBuff.readFile(inFileName);


    size_t dataNumber;
    size_t simNumber;
    inBuff >> simNumber >> dataNumber;

    auto d = NNSimulator::Data<float>::createItem( 
        static_cast<typename NNSimulator::Data<float>::ChildId>( dataNumber ) 
    ); 

    std::ofstream outFile( outFileName );

    while( inBuff >> *d  )
    {
        inBuff >> simNumber;
        inBuff >> dataNumber;
        outFile << d->t << ' ';
        for(int i=0; i<d->nNeurs; ++i)
        {
               outFile << d->VNeurs[i] << ' ' ;
        }
        outFile << std::endl;
    }
}
