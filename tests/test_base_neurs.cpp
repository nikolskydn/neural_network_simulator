#define BOOST_TEST_MODULE NeursTest
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <iostream>
#include <sstream>
#include <vector>
#include <type_traits>
#include "../lib/neursspec.hpp"

#define DEBUG 0

BOOST_AUTO_TEST_SUITE (NeursTest) 

BOOST_AUTO_TEST_CASE (testSpecNeurs)
{

    using ValueType = float;
    //using ValueType = double;
    

    // input data 
    size_t N = 3;
    ValueType t = 0.;
    std::valarray<ValueType> I = { 2., 1., 1. };
    ValueType paramSpec = 5.;
    ValueType VPeak = 3.;
    ValueType VReset = -0.5;
    std::vector<ValueType> V = { -0.5, 2.2, 3.3 };
    std::vector<ValueType> mask = { true, false, false };

    ValueType dt = 0.01;

    // result data for NeursSpec: V_i+=paramSpec*I_i*dt for mask_i=false, V_i=VReset for mask_i=true:w
    size_t resN = 3;
    ValueType rest = 0.01;
    ValueType resParamSpec = 5.;
    ValueType resVPeak = 3.;
    ValueType resVReset = -0.5;
    std::vector<ValueType> resV = { VReset, V[1]+paramSpec*I[1]*dt, V[2]+paramSpec*I[2]*dt };
    std::vector<ValueType> resMask = { false, false, true };


    //the order of the elements: N, t, V, mask, VPeak,  VReset,  paramSpec.
    std::stringstream inBuffer;
    inBuffer << N << ' ' << t << '\t' 
            << V[0]    << ' ' << V[1]    << ' ' << V[2]    << '\t'
            << mask[0] << ' ' << mask[1] << ' ' << mask[2] << '\t'
            << VPeak << ' ' << VReset << '\t'
            << paramSpec << ' ' ;

    #if DEBUG
    std::cout << "\033[36m inBuffer = " << inBuffer.str() << "\033[0m\n";
    #endif

    // make NeursSpec and start perforStepTime()
    std::unique_ptr<NNSimulator::Neurs<ValueType>> neurs;
    neurs = neurs->createItem( NNSimulator::Neurs<ValueType>::ChildId::NeursSpecId );     
    neurs->setCurrents(I);
    inBuffer >> *neurs;
    neurs->performStepTime(dt);
    std::stringstream outBuffer;
    outBuffer << *neurs;
    #if DEBUG
    std::cout << "\033[35m outBuffer = " << outBuffer.str() << "\033[0m\n";
    #endif

    //the order of the elements: N, t, V, mask, VPeak,  VReset,  paramSpec.
    size_t outN;
    outBuffer >> outN;
    BOOST_CHECK( outN == resN ); 

    ValueType outt;
    outBuffer >> outt;
    BOOST_CHECK_CLOSE_FRACTION( outt, rest, std::numeric_limits<ValueType>::epsilon() );

    std::vector<ValueType> outV(N);
    for(size_t i=0; i<N; ++i )
    {
        outBuffer >> outV[i];
        BOOST_CHECK_CLOSE_FRACTION( outV[i], resV[i], std::numeric_limits<ValueType>::epsilon() );    
    }


    std::vector<ValueType> outMask(N);
    for(size_t i=0; i<N; ++i )
    {
        outBuffer >> outMask[i];
        BOOST_CHECK( outMask[i] == resMask[i] ); 
    }

    ValueType outVPeak;
    outBuffer >> outVPeak;
    BOOST_CHECK_CLOSE_FRACTION( outVPeak, resVPeak, std::numeric_limits<ValueType>::epsilon() );

    ValueType outVReset;
    outBuffer >> outVReset;
    BOOST_CHECK_CLOSE_FRACTION( outVReset, resVReset, std::numeric_limits<ValueType>::epsilon() );

    ValueType outParamSpec;
    outBuffer >> outParamSpec;
    BOOST_CHECK_CLOSE_FRACTION( outParamSpec, resParamSpec, std::numeric_limits<ValueType>::epsilon() );

    BOOST_AUTO_TEST_SUITE_END();
}

