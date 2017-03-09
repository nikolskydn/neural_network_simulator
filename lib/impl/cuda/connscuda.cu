/** @addtogroup Conns
 * @{*/

/** @file */


#include <valarray>
#include "connscuda.hpp"

#include <iostream>

namespace NNSimulator {

    __global__ void performStepTimeSpecKernel( const float *dt, const float *paramSpec, const float *V, float *t, float *I ) 
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if(V[i]>10) I[i] += (*paramSpec)*V[i]*(*dt);
        else I[i] *= 0.5;
        *t += *dt;
    }


    //! Полная специализация метода performStepTimeSpec() для float.
    template<> 
    void ConnsImplCuda<float>::performStepTimeSpec(
        const float & dt, 
        const float & paramSpec,
        const std::valarray<float> & V,
        float & t,
        std::valarray<float> & I
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

        performStepTimeSpecKernel<<< 1, n >>>( dtDev, paramSpecDev, VDev, tDev, IDev );

        cudaMemcpy( &t, tDev, size, cudaMemcpyDeviceToHost );
        cudaMemcpy( &I[0], IDev, nSize, cudaMemcpyDeviceToHost );

        cudaFree( dtDev );
        cudaFree( paramSpecDev );
        cudaFree( IDev );
        cudaFree( tDev );
        cudaFree( VDev );

    }

}


/*@}*/
