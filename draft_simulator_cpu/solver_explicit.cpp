#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <type_traits>
#include "../lib/neursspec.hpp"
#include "../lib/connsspec.hpp"

class Solver
{
    std::unique_ptr<NNSimulator::Neurs<float>> neurs;
    std::unique_ptr<NNSimulator::Conns<float>> conns;
    float  dt {0.1};
    NNSimulator::Neurs<float>::ChildId neursId;
    NNSimulator::Conns<float>::ChildId connsId;

public:

    Solver()
    {
    }

    virtual std::istream& read( std::istream& istr ) 
    {
        istr >> dt;
        size_t tmpId;
        istr >> tmpId;
        neursId = static_cast<NNSimulator::Neurs<float>::ChildId>( tmpId );
        neurs = neurs->createItem( neursId ); 
        istr >> *neurs;
        istr >> tmpId;
        connsId = static_cast<NNSimulator::Conns<float>::ChildId>(tmpId);
        conns = conns->createItem( connsId );     
        istr >> *conns;
        neurs->setCurrents( conns->getCurrents() );
        conns->setPotentials( neurs->getPotentials() ); 
        conns->setMasks( neurs->getMasks() ); 
    }

    virtual std::ostream& write( std::ostream& ostr ) const 
    { 
        ostr << dt << ' ';
        ostr << neursId <<  ' ' << *neurs << ' ';
        ostr << connsId << ' ' << *conns << ' ';
    }

    void solve()
    {
        neurs->performStepTimeInit();
        conns->performStepTimeInit();
        for( int i=0; i<5; ++i )
        {
            neurs->performStepTime(dt);
            conns->performStepTime(dt);
        }
        neurs->performStepTimeFinalize();
        conns->performStepTimeFinalize();
    }
};


int main( int argc, char* argv[] )
{
    size_t neursType = 0;
    size_t nNeurs = 3;
    float  t = 0.;
    float  paramSpecNeurs = 10.;
    float VPeak = 3;
    float VReset = -10;
    std::vector<float> V = { 1.1, 2.2, 3.3 };
    std::vector<float> mask = { 0, 0, 1 };

    size_t connsType = 0;
    size_t nConns = 3;
    float paramSpecConns = 0.1;
    std::vector<float> I = { 1.,  2., 3. };

    float dt = 0.2;

    std::stringstream inBuff; 
    inBuff << dt << ' ';
    // neurs
    inBuff << neursType << ' ' << nNeurs << ' ' << t << ' ' 
           << V[0] << ' ' << V[1] << ' ' << V[2] << ' '
           << mask[0] << ' ' << mask[1] << ' ' << mask[2] << ' '
           << VPeak << ' ' << VReset << ' ' << paramSpecNeurs << ' ';

    // conns
    inBuff << connsType << ' ' << nConns << ' ' << t << ' ' 
             << I[0] << ' ' << I[1] << ' ' << I[2] << ' '
             << ' ' << paramSpecConns << ' ';

    std::cout << "\033[35m inBuffer " <<  inBuff.str() << "\033[0m" << std::endl;

    Solver s;
    s.read(inBuff);
    s.solve();
    std::stringstream outBuff;
    s.write( outBuff );
    std::cout << "\033[34m outBuffer " <<  outBuff.str() << "\033[0m" << std::endl;

    return 0;
}
