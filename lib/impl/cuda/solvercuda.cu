#include <valarray>
#include <vector>
#include <cmath>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include "solvercuda.hpp"
#include "../../setting.h"

namespace NNSimulator {

    __global__ void solvePCNNI2003KernelG( 
        const size_t *nN,
        const size_t *nE,
        const float *VP,
        const float *a,
        const float *b,
        const float *c,
        const float *d,
        const float *w, 
        const float *dt,
        const float *te,
        float *V, 
        float *U, 
        bool *m, 
        float *I, 
        float *t,
        float *osc 
    ) 
    {
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if( i<*nN )
        {
            curandState state;
            curand_init(i, 0, 0, &state);
            float ct = *t;
            int cn = 0; 
            while( ct< (*te) )
            {
                if( m[i] ) 
                {
                    V[i] = c[i];
                    U[i] += d[i];
                }
                #ifndef NN_TEST_SOLVERS
                    if( i<(*nE) ) 
                        I[i] = 5.*curand_normal(&state); 
                    else 
                        I[i] = 2.*curand_normal(&state);
                #endif
                for( size_t j=0; j<(*nN); ++j ) if( m[j] ) I[i] += w[i*(*nN)+j];
                V[i] += .5*(*dt)*( .04*V[i]*V[i] + 5.*V[i] + 140. - U[i] + I[i] );
                V[i] += .5*(*dt)*( .04*V[i]*V[i] + 5.*V[i] + 140. - U[i] + I[i] );
                U[i] += (*dt)*a[i]*( b[i]*V[i] - U[i] );
                m[i] = V[i] > (*VP);
                ct += *dt;

                if( i==0  ) osc[ cn*(*nN+1) ] = ct;
                osc[ cn*(*nN+1)+i+1 ] = V[i] ;
                ++cn;

            }
            if( i==0 ) *t = ct;
        } // if i<nN
    }

    //! Полная специализация метода solvePCNNI2003E для float.
    template<> 
    void SolverImplCuda<float>::solvePCNNI2003E
    (
        const size_t & nN,
        const size_t & nE,
        const float & VP,
        const std::valarray<float> & aN,
        const std::valarray<float> & bN,
        const std::valarray<float> & cN,
        const std::valarray<float> & dN,
        const std::valarray<float> & wC,
        const float &  dt,
        const float & te,
        std::valarray<float> & VN,
        std::valarray<float> & UN,
        std::valarray<bool> & mN,
        std::valarray<float> & IN,
        float & t,
        std::deque<std::pair<float,std::valarray<float>>> & og
    ) 
    {
        #ifdef NN_TEST_SOLVERS 
            std::cout << "\033[31;1m    warning: test mode; cuda impl\033[0m\n"; 
        #endif
    
        size_t fnN = nN*sizeof(float);
        size_t nC = nN*nN;
        size_t fnC = nC*sizeof(float);
        size_t bnN = nN*sizeof(bool);
        size_t f1 = sizeof(float);
        size_t s1 = sizeof(size_t);
        size_t nSteps = std::ceil((te-t)/dt);
        size_t fnO = (nN+1)*nSteps*sizeof(float);


        size_t *nND;  cudaMalloc( (void**)&nND, s1 );
        size_t *nED;  cudaMalloc( (void**)&nED, s1 );
        float *VPD;   cudaMalloc( (void**)&VPD, f1 );
        float *aND;   cudaMalloc( (void**)&aND, fnN );
        float *bND;   cudaMalloc( (void**)&bND, fnN );
        float *cND;   cudaMalloc( (void**)&cND, fnN );
        float *dND;   cudaMalloc( (void**)&dND, fnN );
        float *dtD;   cudaMalloc( (void**)&dtD, f1 );
        float *teD;   cudaMalloc( (void**)&teD, f1 );
        float *VND;   cudaMalloc( (void**)&VND, fnN );
        float *UND;   cudaMalloc( (void**)&UND, fnN );
        bool *mND;    cudaMalloc( (void**)&mND, bnN );
        float *IND;   cudaMalloc( (void**)&IND, fnN );
        float *wCD;   cudaMalloc( (void**)&wCD, fnC );
        float *tD;    cudaMalloc( (void**)&tD, f1 );
        float *oscD;  cudaMalloc( (void**)&oscD, fnO );

        cudaMemcpy( nND,  &nN,     s1,   cudaMemcpyHostToDevice );
        cudaMemcpy( nED,  &nE,     s1,   cudaMemcpyHostToDevice );
        cudaMemcpy( VPD,  &VP,     f1,   cudaMemcpyHostToDevice );
        cudaMemcpy( aND,  &aN[0],  fnN,  cudaMemcpyHostToDevice );
        cudaMemcpy( bND,  &bN[0],  fnN,  cudaMemcpyHostToDevice );
        cudaMemcpy( cND,  &cN[0],  fnN,  cudaMemcpyHostToDevice );
        cudaMemcpy( dND,  &dN[0],  fnN,  cudaMemcpyHostToDevice );
        cudaMemcpy( dtD,  &dt,     f1,   cudaMemcpyHostToDevice );
        cudaMemcpy( teD,  &te,     f1,   cudaMemcpyHostToDevice );
        cudaMemcpy( VND,  &VN[0],  fnN,  cudaMemcpyHostToDevice );
        cudaMemcpy( UND,  &UN[0],  fnN,  cudaMemcpyHostToDevice );
        cudaMemcpy( mND,  &mN[0],  bnN,  cudaMemcpyHostToDevice );
        cudaMemcpy( IND,  &IN[0],  fnN,  cudaMemcpyHostToDevice );
        cudaMemcpy( wCD,  &wC[0],  fnC,  cudaMemcpyHostToDevice );
        cudaMemcpy( tD,   &t,      f1,   cudaMemcpyHostToDevice );

        #ifdef TimeDebug
            cudaEvent_t bTime, eTime;
            float cudaTime = .0f;
            cudaEventCreate( &bTime );
            cudaEventCreate( &eTime );
            cudaEventRecord( bTime, 0 );
        #endif
        size_t numBlocks = nN / 512 + 1;
        solvePCNNI2003KernelG<<< numBlocks, 512, 0 >>>
        ( 
            nND, nED, VPD, aND, bND, cND, dND, wCD,  dtD, teD, 
            VND, UND, mND, IND, tD, oscD
        );

        #ifdef TimeDebug
            cudaEventRecord( eTime, 0 );
            cudaEventSynchronize( eTime );
            cudaEventElapsedTime( &cudaTime, bTime, eTime );
            std::cout << "Neurons: \033[31;1m" << nN << "\033[0m\t";
            std::cout << "Connects: \033[31;1m" << nN*nN << "\033[0m\t";
            std::cout << "Steps: \033[31;1m" << nSteps << "\033[0m\t";
            std::cout << "solvePCNNI2003KernelG time: \033[31;1m" << cudaTime << "\033[0m, ms\n";
        #endif

        cudaMemcpy( &t,      tD,   f1,   cudaMemcpyDeviceToHost );
        cudaMemcpy( &VN[0],  VND,  fnN,  cudaMemcpyDeviceToHost );
        cudaMemcpy( &UN[0],  UND,  fnN,  cudaMemcpyDeviceToHost );
        cudaMemcpy( &mN[0],  mND,  bnN,  cudaMemcpyDeviceToHost );
        cudaMemcpy( &IN[0],  IND,  fnN,  cudaMemcpyDeviceToHost );

        float tTmp;
        std::valarray<float> VTmp(nN);
        for(int i=0; i<nSteps; ++i)
        {
            cudaMemcpy( &tTmp,  &oscD[ i*(nN+1) ], f1,  cudaMemcpyDeviceToHost );
            cudaMemcpy( &VTmp[0],  &oscD[ i*(nN+1)+1 ], fnN,  cudaMemcpyDeviceToHost );
            og.push_back(std::pair<float,std::valarray<float>>(tTmp,VTmp));
        }

        cudaFree( nND );
        cudaFree( nED );
        cudaFree( VPD );
        cudaFree( aND );
        cudaFree( bND );
        cudaFree( cND );
        cudaFree( dND );
        cudaFree( dtD );
        cudaFree( teD );
        cudaFree( VND );
        cudaFree( UND );
        cudaFree( mND );
        cudaFree( IND );
        cudaFree( wCD );
        cudaFree( tD );
        cudaFree( oscD );

    }

    //! Полная специализация решателя solveUNN270117
    template<> 
	void SolverImplCuda<float>::solveUNN270117(
            	const size_t& nNeurs,
				const size_t& nNeursExc,
				const size_t& Nastr,
				const float& VNeursPeak,
				const float& VNeursReset,
				const float& dt,
				const float& st,
				float& t,

				const float& Cm,
				const float& g_Na,
				const float& g_K,
				const float& g_leak,
				const float& Iapp,
				const float& E_Na,
				const float& E_K,
				const float& E_L,
				const float& Esyn,
				const float& theta_syn,
				const float& k_syn,

				const float& alphaGlu,
				const float& alphaG,
				const float& bettaG,

				const float& tauIP3,
				const float& IP3ast,
				const float& a2,
				const float& d1,
				const float& d2,
				const float& d3,
				const float& d5,

				const float& dCa,
				const float& dIP3,
				const float& c0,
				const float& c1,
				const float& v1,
				const float& v4,
				const float& alpha,
				const float& k4,
				const float& v2,
				const float& v3,
				const float& k3,
				const float& v5,
				const float& v6,
				const float& k2,
				const float& k1,

				const float& IstimAmplitude,
				const float& IstimFrequency,
				const float& IstimDuration,
				std::valarray<float>& nextTimeEvent,
				std::valarray<float>& Istim,

				const std::valarray<float>& wConns,
				const std::valarray<float>& wAstrNeurs,
				const std::valarray<bool>& astrConns,

				std::valarray<float>& VNeurs,
				std::valarray<bool>& mNeurs,
				std::valarray<float>& INeurs,
				std::valarray<float>& m,
				std::valarray<float>& h,
				std::valarray<float>& n,
				std::valarray<float>& G,

				std::valarray<float>& Ca,
				std::valarray<float>& IP3,
				std::valarray<float>& z,

				std::deque<std::pair<float,std::valarray<float>>>& oscillograms
            ) 
			{
				std::cerr<<"ERROR in SolverImplCuda<float>::solveUNN270117(...)"<<std::endl;
				std::cerr<<"\tThere is no CUDA implementation for model UNN270117"<<std::endl<<std::endl;
				throw;
			}

}

