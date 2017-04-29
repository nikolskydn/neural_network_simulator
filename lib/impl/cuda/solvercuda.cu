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
        size_t i = threadIdx.x + blockIdx.x * blockDim.x;
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
    }

    //! Полная специализация метода solvePCNNI2003E для float.
    template<> 
    void SolverImplCuda<float>::solvePCNNI2003E
    (
        const size_t & nN,
        const size_t & nE,
        const float & VP,
        const std::valarray<float> aN,
        const std::valarray<float> bN,
        const std::valarray<float> cN,
        const std::valarray<float> dN,
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
        solvePCNNI2003KernelG<<< 1, nN >>>
        ( 
            nND, nED, VPD, aND, bND, cND, dND, wCD,  dtD, teD, 
            VND, UND, mND, IND, tD, oscD
        );

        #ifdef TimeDebug
            cudaEventRecord( eTime, 0 );
            cudaEventSynchronize( eTime );
            cudaEventElapsedTime( &cudaTime, bTime, eTime );
            std::cout << "Neurons: " << nN << std::endl;
            std::cout << "Connects: " << nN*nN << std::endl;
            std::cout << "Steps: " << nSteps << std::endl;
            std::cout << "solvePCNNI2003KernelG time: " << cudaTime << ", ms\n\n";
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

}

