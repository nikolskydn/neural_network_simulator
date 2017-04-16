#include <iostream>
#include <sstream>
#include <fstream>
#include <type_traits>
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
    SolversBuffer inBuff;
    inBuff.readFile(inFileName);

    size_t dataNumber;
    size_t simNumber;
    inBuff >> simNumber >> dataNumber;
    inBuff << simNumber << dataNumber;

    auto d = NNSimulator::Data<float>::createItem( 
            static_cast<typename NNSimulator::Data<float>::ChildId>( dataNumber ) 
    ); 


    while( inBuff >> *d  )
    {
        inBuff >> simNumber;
        inBuff >> dataNumber;
        for(int i=0; i<d->nNeurs; ++i)
        {
           if( d->mNeurs[i] ) 
               std::cout << d->t << ' ' << i << std::endl;
        }
    }



}
