#define BOOST_TEST_MODULE ConnsTest
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <iostream>
#include <sstream>
#include <vector>
#include <type_traits>
#include "../lib/connsspec.hpp"



BOOST_AUTO_TEST_SUITE (ConnsTest) 

BOOST_AUTO_TEST_CASE (testSpecConns)
{

    using ValueType = float;
    //using ValueType = double;
    

    // in data
    size_t inN = 3;
    ValueType in_t = 0.;
    ValueType dt = 0.1;
    ValueType paramSpec = 0.1;
    std::vector<ValueType> inI = { 1.,  2., 3. };
    std::valarray<ValueType> V = { 21., 1., 20.};
    std::valarray<bool> mask = { 1, 0, 1};
    std::vector<float> W = {  0, 1., 0,   0, 0, 2.,   3., 0, 0 };

    // out data
    std::vector<ValueType> resI = { 1.21,  1.,  3.2 };

    std::stringstream inBuffer;
    inBuffer << inN << ' ' << in_t << ' ' 
             << inI[0] << ' ' << inI[1] << ' ' << inI[2] << ' '
             << W[0] << ' ' << W[1] << ' ' << W[2] << ' '
             << W[3] << ' ' << W[4] << ' ' << W[5] << ' '
             << W[6] << ' ' << W[7] << ' ' << W[8] << ' '
             << paramSpec;

    std::unique_ptr<NNSimulator::Conns<ValueType>> conns;
    conns = conns->createItem( NNSimulator::Conns<ValueType>::ChildId::ConnsSpecId );     

    conns->setPotentials(V); 
    conns->setMasks(mask); 
    inBuffer >> *conns;
    conns->performStepTimeInit();
    conns->performStepTime(dt);
    conns->performStepTimeFinalize();

    std::stringstream outBuffer;
    outBuffer << *conns;

    size_t outN;
    outBuffer >> outN;
    BOOST_CHECK( inN = outN ); 

    ValueType out_t;
    outBuffer >> out_t;
    BOOST_CHECK_CLOSE_FRACTION ( out_t, in_t+dt, std::numeric_limits<ValueType>::epsilon() );

    std::vector<ValueType> outI(inN);

    for(size_t i=0; i<inN; ++i )
    {
        outBuffer >> outI[i];
        BOOST_CHECK_CLOSE_FRACTION ( outI[i], resI[i], std::numeric_limits<ValueType>::epsilon() );    
    }

    std::vector<ValueType> outW(inN*inN);

    for(size_t i=0; i<inN*inN; ++i )
    {
        outBuffer >> outW[i];
        BOOST_CHECK_CLOSE_FRACTION ( outW[i], W[i], std::numeric_limits<ValueType>::epsilon() );    
    }




    BOOST_AUTO_TEST_SUITE_END( )
}

