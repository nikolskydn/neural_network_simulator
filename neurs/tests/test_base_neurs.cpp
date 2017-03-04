#define BOOST_TEST_MODULE NeursTest
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <iostream>
#include <sstream>
#include <vector>
#include <type_traits>
#include "../neursspec.hpp"



BOOST_AUTO_TEST_SUITE (NeursTest) 

BOOST_AUTO_TEST_CASE (testSpecNeurs)
{

    using ValueType = float;
    //using ValueType = double;
    

    size_t inN = 3;
    ValueType in_t = 0.;
    ValueType inI = 2.;
    ValueType dt = 0.1;
    ValueType paramSpec = 0.1;

    std::vector<ValueType> inV = { 1.1, 2.2, 3.3 };

    std::stringstream inBuffer;
    inBuffer << inN << ' ' << in_t << ' ' 
             << inV[0] << ' ' << inV[1] << ' ' << inV[2] << ' '
             << paramSpec;

    std::valarray<ValueType> I( inI, inN );

    std::unique_ptr<NNSimulator::Neurs<ValueType>> neurs;
    neurs = neurs->createItem( NNSimulator::Neurs<ValueType>::ChildId::NeursSpecId );     

    neurs->setCurrents(I);
    inBuffer >> *neurs;
    neurs->performStepTime(dt);

    std::stringstream outBuffer;
    outBuffer << *neurs;

    size_t outN;
    outBuffer >> outN;
    BOOST_CHECK( inN = outN ); 

    ValueType out_t;
    outBuffer >> out_t;
    BOOST_CHECK_CLOSE_FRACTION ( out_t, in_t+dt, std::numeric_limits<ValueType>::epsilon() );

    std::vector<ValueType> outV(inN);

    for(size_t i=0; i<inN; ++i )
    {
        outBuffer >> outV[i];
        BOOST_CHECK_CLOSE_FRACTION ( outV[i], inV[i]+0.1*inI*dt, std::numeric_limits<ValueType>::epsilon() );    }



    BOOST_AUTO_TEST_SUITE_END( )
}

