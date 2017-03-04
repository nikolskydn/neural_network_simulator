/** @addtogroup Neurs
 * @{*/

/** @file */


#include <valarray>
#include "neurscuda.hpp"

#include <iostream>

namespace NNSimulator {

    __global__ void performStepTimeSpecKernel( const float *dt, const float *paramSpec, const float *I, float *t, float *V ) 
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        V[i] += (*paramSpec)*I[i]*(*dt);
        *t += *dt;
    }


    //! Полная специализация метода performStepTimeSpec() для float.
    template<> 
    void NeursImplCuda<float>::performStepTimeSpec(
        const float & dt, 
        const float & paramSpec,
        const std::valarray<float> & I,
        float & t,
        std::valarray<float> & V 
    ) 
    {
        size_t n = I.size();
        size_t nSize = n*sizeof(float);
        size_t size = sizeof(float);

        float *dtDev;
        float *tDev;
        float *paramSpecDev;
        float *IDev;
        float *VDev;

        cudaMalloc( (void**)&dtDev, size);
        cudaMalloc( (void**)&tDev, size);
        cudaMalloc( (void**)&paramSpecDev, size);
        cudaMalloc( (void**)&IDev, nSize);
        cudaMalloc( (void**)&VDev, nSize);

        cudaMemcpy( dtDev, &dt, size, cudaMemcpyHostToDevice );
        cudaMemcpy( tDev, &t, size, cudaMemcpyHostToDevice );
        cudaMemcpy( paramSpecDev, &paramSpec, size, cudaMemcpyHostToDevice );
        cudaMemcpy( IDev, &I[0], nSize, cudaMemcpyHostToDevice );
        cudaMemcpy( VDev, &V[0], nSize, cudaMemcpyHostToDevice );

        performStepTimeSpecKernel<<< 1, n >>>( dtDev, paramSpecDev, IDev, tDev, VDev );

        cudaMemcpy( &t, tDev, size, cudaMemcpyDeviceToHost );
        cudaMemcpy( &V[0], VDev, nSize, cudaMemcpyDeviceToHost );

        cudaFree( dtDev );
        cudaFree( paramSpecDev );
        cudaFree( IDev );
        cudaFree( tDev );
        cudaFree( VDev );

    }

}

