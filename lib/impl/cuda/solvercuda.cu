#include <valarray>
#include <cmath>
#include "solvercuda.hpp"
#include <curand_kernel.h>

#include <iostream>

#define CudaTimeDebug 1

namespace NNSimulator {

    __global__ void solvePCNNI2003KernelG( 
        const int *nN,
        const int *nE,
        const float *VP,
        const float *a,
        const float *b,
        const float *c,
        const float *d,
        const float *dt,
        const float *te,
        float *V, 
        float *U, 
        bool *m, 
        float *I, 
        float *w, 
        float *t
        //, 
        //float *spk, 
        //float *osc 
    ) 
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        curandState s;
        curand_init(i, 0, 0, &s);
        int ct = *t;
        while( ct< (*te) )
        {
            if( m[i] ) 
            {
                V[i] = c[i];
                U[i] += d[i];
            }
            if( i<(*nE) ) 
                I[i] = 5.*curand_uniform(&s); 
            else 
                I[i] = 2.*curand_uniform(&s);
            for( size_t j=0; j<(*nN); ++j ) if( m[j] ) I[i] += w[i*(*nN)+j];
            V[i] += .5*(*dt)*( .04*V[i]*V[i] + 5.*V[i] + 140. - U[i] + I[i] );
            V[i] += .5*(*dt)*( .04*V[i]*V[i] + 5.*V[i] + 140. - U[i] + I[i] );
            U[i] += (*dt)*a[i]*( b[i]*V[i] - U[i] );
            m[i] = V[i] > (*VP);
            ct += *dt;
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
        const float & VR,
        const std::valarray<float> aN,
        const std::valarray<float> bN,
        const std::valarray<float> cN,
        const std::valarray<float> dN,
        const float &  dt,
        const float & te,
        std::valarray<float> & VN,
        std::valarray<float> & UN,
        std::valarray<bool> & mN,
        std::valarray<float> & IN,
        std::valarray<float> & wC,
        float & t,
        std::vector<std::pair<size_t,float>> & spk,
        std::vector<std::pair<size_t,std::valarray<float>>> & osc
    ) 
    {

        size_t fnN = nN*sizeof(float);
        size_t nC = nN*nN;
        size_t fnC = nC*sizeof(float);
        size_t bnN = nN*sizeof(bool);
        size_t f1 = sizeof(float);
        size_t i1 = sizeof(int);
        //size_t fnS = 100000*sizeof(float);
        //size_t fnO = 100000*sizeof(float);

        int *nND;   cudaMalloc( (void**)&nND, i1 );
        int *nED;   cudaMalloc( (void**)&nED, i1 );
        float *VPD;   cudaMalloc( (void**)&VPD, f1 );
        //float *VRD;   cudaMalloc( (void**)&VRD, f1 );
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
        //float *spkD;  cudaMalloc( (void**)&spkD, fnS );
        //float *oscD;  cudaMalloc( (void**)&oscD, fnO );

        int tmpnN = static_cast<int>(nN);
        int tmpnE = static_cast<int>(nE);
        cudaMemcpy( nND,  &tmpnN,     i1,   cudaMemcpyHostToDevice );
        cudaMemcpy( nED,  &tmpnE,     i1,   cudaMemcpyHostToDevice );
        cudaMemcpy( VPD,  &VP,     f1,   cudaMemcpyHostToDevice );
        //cudaMemcpy( VRD,  &VR,     size,   cudaMemcpyHostToDevice );
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
        //cudaMemcpy( spkD, &spk[0], fnS,  cudaMemcpyHostToDevice );
        //cudaMemcpy( oscD, &osc[0], fnO,  cudaMemcpyHostToDevice );

#if CudaTimeDebug>0
        cudaEvent_t bTime, eTime;
        float cudaTime = .0f;
        cudaEventCreate( &bTime );
        cudaEventCreate( &eTime );
        cudaEventRecord( bTime, 0 );
#endif
        solvePCNNI2003KernelG<<< 1, nN >>>
        ( 
            nND, nED, VPD, aND, bND, cND, dND, dtD, teD, 
            VND, UND, mND, IND, wCD, tD /*, spkD, oscD */
        );

#if CudaTimeDebug>0
        cudaEventRecord( eTime, 0 );
        cudaEventSynchronize( eTime );
        cudaEventElapsedTime( &cudaTime, bTime, eTime );
        std::cout << "Neurons: " << nN << std::endl;
        std::cout << "Connects: " << nN*nN << std::endl;
        std::cout << "Steps: " << std::ceil(te/dt) << std::endl;
        std::cout << "solvePCNNI2003KernelG time: " << cudaTime << ", ms\n\n";
#endif

        cudaMemcpy( &t,      tD,   f1,   cudaMemcpyDeviceToHost );
        cudaMemcpy( &VN[0],  VND,  fnN,  cudaMemcpyDeviceToHost );
        cudaMemcpy( &mN[0],  mND,  bnN,  cudaMemcpyDeviceToHost );
        cudaMemcpy( &IN[0],  IND,  fnN,  cudaMemcpyDeviceToHost );


        cudaFree( nND );
        cudaFree( nED );
        cudaFree( VPD );
        //cudaFree( VRD );
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
        //cudaFree( spkD );
        //cudaFree( oscD );

    }

}

