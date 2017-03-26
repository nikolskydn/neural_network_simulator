#define BOOST_TEST_MODULE NeursTest
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <iostream>
#include <sstream>
#include <vector>
#include <type_traits>
#include "../lib/solverexpliciteulerspec.hpp"
#include "../lib/formatstream.hpp"

#define DEBUG 1

BOOST_AUTO_TEST_SUITE (SolverTest) 

BOOST_AUTO_TEST_CASE (testForExplicitSolverWithSpecElems)
{
    size_t neursType = 0;
    size_t nNeurs = 3;
    float  paramSpecNeurs = 10.;
    float VPeak = 3;
    float VReset = -10;
    std::vector<float> V = { 1.1, 2.2, 3.3 };
    std::vector<int> mask = { 0, 0, 1 };

    size_t connsType = 0;
    size_t nConns = 3;
    float paramSpecConns = 0.1;
    std::vector<float> I = { 1.,  2., 3. };
    std::vector<float> W = {  0, 1., 0,   0, 0, 2.,   3., 0, 0 };

    float  t = 0.;
    float dt = 0.2;
    float simulationTime = dt/2.;

    std::stringstream inBuff;
    FormatStream inFBuff( inBuff );

    // solver
    size_t numberSolver = 0;
    inFBuff << numberSolver << t << simulationTime << dt;
    // neurs
    inFBuff << nNeurs  
            << V[0] << V[1] << V[2] 
            << mask[0] << mask[1] << mask[2] 
            << VPeak  << VReset;

    // conns
    inFBuff << I[0] << I[1] << I[2]
            << W[0] << W[1] << W[2]
            << W[3] << W[4] << W[5]
            << W[6] << W[7] << W[8];

    // spec
    inFBuff << paramSpecNeurs << paramSpecConns;

    #if DEBUG 
        std::cout << "\033[31;1m";
        FormatStream f(std::cout, 5);
        // solver:
        f << "nS" << "t" << "sT" << "dt" ;
        // neurs :
        f << "nN";
        f<< "V" ; for(int i=1; i<nNeurs; ++i ) f << ' ';
        f<< "m" ; for(int i=1; i<nNeurs; ++i ) f << ' ';
        f << "VP" << "VR" ;
        // conns:
        f<< "I" ; for(int i=1; i<nNeurs; ++i ) f << ' ';
        f<< "W" ; for(int i=1; i<nNeurs*nNeurs; ++i ) f << ' ';
        f << "PN" << "PC";
        std::cout << "\033[0m" << std::endl;
        std::cout << "\033[31m" <<  inBuff.str() << "\033[0m" << std::endl;
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
        std::cout << "\033[34m";
        f <<  outNumberSolver;
        std::cout << outBuff.str() << "\033[0m" << std::endl;
    #endif

    // ------------    test Neurs params ------------------------
    /*
     * NeursSpec
     * {
     *     V[mask] = VReset
     *     V[!mask] += neursParamSpec*dt*I[!mask]
     *     mask = V>VPeak
     * }
     */
    float outt;
    outBuff >> outt;
    BOOST_CHECK_CLOSE_FRACTION( outt, t+dt, std::numeric_limits<float>::epsilon() );

    float outSimulationTime;
    outBuff >> outSimulationTime;
    BOOST_CHECK_CLOSE_FRACTION( outSimulationTime, simulationTime, std::numeric_limits<float>::epsilon() );

    float outDt;
    outBuff >> outDt;
    BOOST_CHECK_CLOSE_FRACTION( outDt, dt, std::numeric_limits<float>::epsilon() );

    size_t outNNeurs;
    outBuff >> outNNeurs;
    BOOST_CHECK( outNNeurs == nNeurs ); 

    std::vector<float> outV(nNeurs);
    for(size_t i=0; i<nNeurs; ++i ){
        outBuff >> outV[i];
        if(mask[i]) V[i] = VReset;
        else V[i] += dt*paramSpecNeurs*I[i];
        BOOST_CHECK_CLOSE_FRACTION( outV[i], V[i], std::numeric_limits<float>::epsilon() );
    }

    std::vector<int> outMask(nNeurs);
    for(size_t i=0; i<nNeurs; ++i ){
        outBuff >> outMask[i];
        if(V[i]>VPeak) mask[i] = true; 
        else mask[i] = false;
        BOOST_CHECK( outMask[i] == mask[i] ); 
    }

    float outVPeak;
    outBuff >> outVPeak;
    BOOST_CHECK_CLOSE_FRACTION( outVPeak, VPeak, std::numeric_limits<float>::epsilon() );

    float outVReset;
    outBuff >> outVReset;
    BOOST_CHECK_CLOSE_FRACTION( outVReset, VReset, std::numeric_limits<float>::epsilon() );

    // ------------    test ConnsSpec params ------------------------
    /*
     * ConnsSpec
     * {
     *     I[!mask] *= 0.5
     *     I[mask] += connsParamSpec*dt*V[mask]
     * }
     */
    
    std::vector<float> outI(nNeurs);
    for(size_t i=0; i<nNeurs; ++i ){
        outBuff >> outI[i];
        if(mask[i]) I[i] += dt*paramSpecConns*V[i]; 
        else I[i] *= .5f;
        BOOST_CHECK_CLOSE_FRACTION( outI[i], I[i], std::numeric_limits<float>::epsilon() );
    }

    std::vector<size_t> outW(nNeurs*nNeurs);
    for(size_t i=0; i<nNeurs*nNeurs; ++i ){
        outBuff >> outW[i];
        BOOST_CHECK( outW[i] == W[i] ); 
    }

    float outParamSpecNeurs;
    outBuff >> outParamSpecNeurs;
    BOOST_CHECK_CLOSE_FRACTION( outParamSpecNeurs, paramSpecNeurs, std::numeric_limits<float>::epsilon() );

    float outParamSpecConns;
    outBuff >> outParamSpecConns;
    BOOST_CHECK_CLOSE_FRACTION( outParamSpecConns, paramSpecConns, std::numeric_limits<float>::epsilon() );


    BOOST_AUTO_TEST_SUITE_END();
}

