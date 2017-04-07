/** @addtogroup Plugin
 * @{*/

/** @file */

#ifndef _NNetworkSimulatorSolver_sBufferNDN2017_
#define _NNetworkSimulatorSolver_sBufferNDN2017_

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

class SolversBuffer : public std::stringstream 
{
    public:

        explicit SolversBuffer()   {}
    
        void readFile( std::string & fileName  )
        {
           std::stringstream rawData;
           std::ifstream src( fileName );
           if( src.is_open() ) rawData << src.rdbuf();
           else
           {
               std::cerr << "\033[1;31;40merror: \033[1;33;40mopening file '" << fileName << "'\033[0m\n";
               throw;
           }
           std::string rawLine;
           while( std::getline( rawData, rawLine) )
           {
               if( rawLine[0] != '#' && rawLine[0] != '\n' )
               {
                    *this << rawLine << ' ';
               }
           }
        }

        void clean(){ str(std::string()); }
};
#endif

/*@}*/
