#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <valarray>
#include <type_traits>
#include "../lib/data.hpp"
#include "../lib/datapcnni2003.hpp"
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

    size_t sNum, dNum, nNeurs, nNeursExc;
    float t, dt, tEnd, dtDump, VPeak, VReset;
    inBuff >> sNum;
    inBuff >> dNum;
    inBuff >> nNeurs;
    inBuff >> nNeursExc;
    inBuff >> t;
    inBuff >> tEnd;
    inBuff >> dt;
    inBuff >> dtDump;
    inBuff >> VPeak;
    inBuff >> VReset;

    auto d = std::make_unique<NNSimulator::DataPCNNI2003<float>>();

    d->t = t;
    d->dt = dt;
    d->tEnd = tEnd;
    d->dtDump = dtDump;

    d->nNeurs = nNeurs;
    d->nNeursExc = nNeursExc;
    d->VNeursPeak = VPeak;
    d->VNeursReset = VReset;
    d->VNeurs.resize(nNeurs,VReset);
    d->mNeurs.resize(nNeurs,false);

    d->INeurs.resize(nNeurs);
    size_t nConns = nNeurs*nNeurs;
    d->wConns.resize(nConns);

    d->UNeurs.resize(nNeurs);
    d->aNeurs.resize(nNeurs);
    d->bNeurs.resize(nNeurs);
    d->cNeurs.resize(nNeurs);
    d->dNeurs.resize(nNeurs);


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> udis(0,1);
    std::normal_distribution<> ndis;

    std::valarray<float> re(nNeursExc);
    std::valarray<float> ri(nNeurs-nNeursExc);

    //for(auto & e: re) e = udis(gen); 
    //for(auto & e: ri) e = udis(gen); 

    for( size_t i=0 ; i<nNeurs; ++ i ) 
        if( i< nNeursExc)
            d->INeurs[i] = 5*ndis(gen);
        else
            d->INeurs[i] = 2*ndis(gen);

    for( size_t i=0 ; i<nNeurs; ++i ) 
        for( size_t j=0 ; j<nNeurs; ++j ) 
            if( j< nNeursExc)
                d->wConns[i*nNeurs+j] = 0.5*udis(gen);
            else
                d->wConns[i*nNeurs+j] = -udis(gen);
    
    for( size_t i=0 ; i<nNeurs; ++ i ) 
        if( i< nNeursExc)
        {
            d->aNeurs[i] = 0.02*ndis(gen);
            d->bNeurs[i] = 0.2*ndis(gen);
            d->cNeurs[i] = -65+15*ndis(gen)*ndis(gen);
            d->dNeurs[i] = 8-6*ndis(gen)*ndis(gen);
        }
        else
        {
            d->aNeurs[i] = 0.02+0.08*ndis(gen);
            d->bNeurs[i] = 0.25-0.05*ndis(gen);
            d->cNeurs[i] = -65;
            d->dNeurs[i] = 2;
        }

    d->UNeurs = d->bNeurs * d->VNeurs;

    std::cout << sNum << ' ' << dNum << ' ';
    std::cout << *d;
}
