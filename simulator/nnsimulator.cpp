#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <regex>
#include <sstream>
#include "../lib/solver.hpp"
#include "../lib/formatstream.hpp"
#include "../lib/solversbuffer.hpp"
#include "../tools/binarybuffer.h"
#include "../tools/workwithfiles.h"
#include "../tools/systemtimestopwatch.h"

#define DEBUG 0

int main(int argc, char* argv[])
{
	SystemTimeStopwatch stopwatch;
	stopwatch.start();

	std::cout << "arguments:" << std::endl;
	std::cout << "argc = " << argc << std::endl;
	for(int i=0; i < argc; ++i){
		std::cout << "argv[" << i << "] = " << argv[i] << std::endl;
	}
	std::cout << std::endl;

    #if DEBUG
        std::cout << "mode DEBUG = " << DEBUG << std::endl;
    #endif

    if( argc < 2 ){
	std::cout << "few arguments. You must give a name of an IN-file (.sin or .bsin) as the first argument" << std::endl;
        exit(1);
    }

    std::string inFileName, outFileName, spksFileName, oscsFileName, casFileName;

    bool isBin = false;
    bool isBinRead = false;

    {
        inFileName = argv[1];

        std::string vs1= ".sin";
        bool finded;
        isBinRead = check_file_extension( vs1, inFileName, finded );
        std::regex exts( "\\" + vs1 + "$" );
        std::string vsOut = ".sout";

        if (!finded){
            std::cout << "Error: the extension must be 'sin' or 'bsin' for in file.\n";
            exit(1);
        }

	if (argc == 2){
	    if (isBinRead){			// if in file is binary, then out file is also binary. ( for filename in argv[1]. Not true for ID in argv[1] )
		    isBin = true;
		    vsOut = ".bsout";
	    }
	}
	else{
	    if (argv[2] == "0"){
		isBin = false;
	    }
	    else if (argv[2] == "1"){
		isBin = true;
		vsOut = ".bsout";
	    }
	    else{
		std::cout << "Error: the second argument must be 0 (outfile is not binary) or 1 (outfile is binary)" << std::endl;
		exit(-1);
	    }
	}

	outFileName = std::regex_replace(
	        inFileName,
	        exts,
	        vsOut
	);
	std::string vs3= ".spks";
	spksFileName = std::regex_replace(
	        inFileName,
	        exts,
	        vs3
	);
	std::string vs4= ".oscs";
	oscsFileName = std::regex_replace(
	        inFileName,
	        exts,
	        vs4
	);
	std::string vs5= ".cas";
	casFileName = std::regex_replace(
	        inFileName,
	        exts,
	        vs5
	);
    }
    #if DEBUG
        std::cout << "inFileName:   \033[35;3m" << inFileName   << "\033[0m\n";
	    std::cout << "outFileName:  \033[35;3m" << outFileName  << "\033[0m\n";
	    std::cout << "spksFileName: \033[35;3m" << spksFileName << "\033[0m\n";
	    std::cout << "oscsFileName: \033[35;3m" << oscsFileName << "\033[0m\n";
	    std::cout << "casFileName:  \033[35;3m" << casFileName  << "\033[0m\n";
    #endif

    SolversBuffer inBuff;

    inBuff.useOnlyRawData( isBinRead );
    inBuff.readFile(inFileName);
    #if DEBUG > 1 
        std::cout << "\n\033[32;1mSolverPCNN = { ";
        std::cout << "\033[32;2m" <<  inBuff.str() << " \033[32;1m}\033[0m" << std::endl;
    #endif

    size_t outNumberSolver;
    std::stringstream forCheckSS;
    std::ofstream outFileStream( outFileName );
    forCheckSS.str( inBuff.str() );

    if (isBinRead){		// checking
    	BinaryBufferInS binin( forCheckSS );
    	if ( !(binin >> outNumberSolver) ){
    		std::cerr << "ERROR: Small amount of params in the infile for a simulator. Can\'t read a number of solver - first parameter" << std::endl << std::endl;
    		outFileStream << "ERROR: Small amount of params in the infile for a simulator. Can\'t read a number of solver - first parameter" << std::endl << std::endl;
    		throw;
    	}
    }
    else{
    	if ( !(forCheckSS >> outNumberSolver) ){
    		std::cerr << "ERROR: Small amount of params in the infile for a simulator. Can\'t read a number of solver - first parameter" << std::endl << std::endl;
    		outFileStream << "ERROR: Small amount of params in the infile for a simulator. Can\'t read a number of solver - first parameter" << std::endl << std::endl;
    		throw;
    	}
    }

    if (isBinRead){		// numer of solver
    	BinaryBufferInS binin( inBuff );
    	binin >> outNumberSolver;
    }
    else{
    	inBuff >> outNumberSolver;
    }

    auto s = NNSimulator::Solver<float>::createItem( 
            static_cast<typename NNSimulator::Solver<float>::ChildId>( outNumberSolver ) 
    );
    s->setIsBinaryWriteFlag( isBin );
    s->setIsBinaryReadFlag( isBinRead );

    // checking
    std::pair<bool,std::string> checkObj = s->checkFile( forCheckSS );
    if ( !checkObj.first ){
    	std::cerr << "ERROR in the infile for simulator: " << std::endl << checkObj.second << std::endl << std::endl;
    	outFileStream << "ERROR in the infile for simulator: " << std::endl << checkObj.second << std::endl << std::endl;
    	throw;
    }
    else{
    	std::cout << "The infile for simulator checked. OK" << std::endl;
    }

    // reading
    s->read( inBuff );

    s->solve( std::move(outFileStream) );
    //s->solve( std::move(std::cout) );

    auto citsSpikes = s->getSpikes();
    std::ofstream spksFileStream(spksFileName);
    spksFileStream << "# Spikes" << std::endl;
    spksFileStream << "# time, ms" << std::endl;
    spksFileStream << "# Neuron No." << std::endl;
    std::copy
    (
        citsSpikes.first,
        citsSpikes.second,
        std::ostream_iterator<std::pair<size_t,float>>( spksFileStream, "\n" )
    );
     
    auto citsOscillograms = s->getOscillograms();
    std::ofstream oscsFileStream( oscsFileName );
    oscsFileStream << "# Voltage on neurons" << std::endl;
    oscsFileStream << "# time, ms" << std::endl;
    oscsFileStream << "# V" << std::endl;
    std::copy
    (
        citsOscillograms.first,
        citsOscillograms.second,
        std::ostream_iterator<std::pair<float,std::valarray<float>>>( oscsFileStream, "\n" )
    );


    #if DEBUG > 1
        std::stringstream outBuff;
        s->write( outBuff );
        std::cout << "\033[34m";
        std::cout << outBuff.str() << "\033[0m" << std::endl;
    #endif

	auto dtFull = stopwatch.lookTimeSegmentFromStart();
	std::string dtFullStr = convertTimeIntoString( dtFull.count() );
	std::cout << "dtFull = " << dtFullStr << std::endl;

}
