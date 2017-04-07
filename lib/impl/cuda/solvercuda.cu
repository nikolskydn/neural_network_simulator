/** @addtogroup Solver
 * @{*/

/** @file */

#include <valarray>
#include "solvercuda.hpp"

#include <iostream>

namespace NNSimulator {

    __global__ void solveTestKernelG(  
        const float *VP,
        const float *VR,
        const float *dt,
        const float *st,
        const float *np,
        const float *cp,
        float *V,
        bool *m,
        float *I,
        float *w,
        float *t
    ) 
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if( m[i] ) V[i] = *VR;
        else V[i] += (*np)*I[i]*(*dt);
        m[i] = V[i] > *VP;
        if(V[i]>(*VP)) I[i] += (*cp)*V[i]*(*dt);
        else I[i] *= 0.5;
        *t += *dt;

    }

    //! Полная специализация метода solveTest для float.
    template<> 
    void SolverImplCuda<float>::solveTest(
        const size_t & nN, // nNeurs
        const float & VP, // VPeak
        const float & VR, // VReset
        const float &  dt,
        const float & st, // simulationTime
        const float & np, // neuronsParamSpec
        const float & cp, // connectsParamSpec
        std::valarray<float> & V,
        std::valarray<bool> & m, // mask
        std::valarray<float> & I,
        std::valarray<float> & w, // weight
        float & t
    ) 
    {
        size_t nSize = nN*sizeof(float);
        size_t bSize = nN*sizeof(bool);
        size_t size = sizeof(float);

        float *VPD;
        float *VRD;
        float *dtD;
        float *stD;
        float *npD;
        float *cpD;
        float *VD;  
        bool *mD;  
        float *ID;  
        float *wD; 
        float *tD;

        cudaMalloc( (void**)&VPD, size);
        cudaMalloc( (void**)&VRD, size);
        cudaMalloc( (void**)&dtD, size);
        cudaMalloc( (void**)&stD, size);
        cudaMalloc( (void**)&npD, size);
        cudaMalloc( (void**)&cpD, size);
        cudaMalloc( (void**)&VD, nSize);
        cudaMalloc( (void**)&mD, bSize);
        cudaMalloc( (void**)&ID, nSize);
        cudaMalloc( (void**)&wD, nSize*nSize);
        cudaMalloc( (void**)&tD, size);

        cudaMemcpy( VPD, &VP, size, cudaMemcpyHostToDevice );
        cudaMemcpy( VRD, &VR, size, cudaMemcpyHostToDevice );
        cudaMemcpy( dtD, &dt, size, cudaMemcpyHostToDevice );
        cudaMemcpy( stD, &st, size, cudaMemcpyHostToDevice );
        cudaMemcpy( npD, &np, size, cudaMemcpyHostToDevice );
        cudaMemcpy( cpD, &cp, size, cudaMemcpyHostToDevice );
        cudaMemcpy( VD, &V[0], nSize, cudaMemcpyHostToDevice );
        cudaMemcpy( mD, &m[0], bSize, cudaMemcpyHostToDevice );
        cudaMemcpy( ID, &I[0], nSize, cudaMemcpyHostToDevice );
        cudaMemcpy( wD, &w[0], nSize*nSize, cudaMemcpyHostToDevice );
        cudaMemcpy( tD, &t, size, cudaMemcpyHostToDevice );

        solveTestKernelG<<< 1, nN >>>( VPD, VRD, dtD, stD, npD, cpD, VD, mD,  ID,  wD, tD);

        cudaMemcpy( &t, tD, size, cudaMemcpyDeviceToHost );
        cudaMemcpy( &V[0], VD, nSize, cudaMemcpyDeviceToHost );
        cudaMemcpy( &m[0], mD, bSize, cudaMemcpyDeviceToHost );
        cudaMemcpy( &I[0], ID, nSize, cudaMemcpyDeviceToHost );

        cudaFree( VPD );
        cudaFree( VRD );
        cudaFree( dtD );
        cudaFree( stD );
        cudaFree( npD );
        cudaFree( cpD );
        cudaFree( VD );
        cudaFree( mD );
        cudaFree( ID );
        cudaFree( wD );
        cudaFree( tD );

    }

}

