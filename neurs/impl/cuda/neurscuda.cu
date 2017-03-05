/** @addtogroup Neurs
 * @{*/

/** @file */


#include <valarray>
#include "neurscuda.hpp"

#include <iostream>

namespace NNSimulator {

    __global__ void performStepTimeSpecKernel( 
        const float *dt, 
        const float *I, 
        const float *VPeak, 
        const float *VReset, 
        const float *paramSpec, 
        float *t, 
        float *V,
        bool *mask 
    ) 
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if( mask[i] )
        {
            V[i] = *VReset;
        } else {
            V[i] += (*paramSpec)*I[i]*(*dt);
        }
        *t += *dt;
        mask[i] = V[i] > *VPeak;
    }


    //! Полная специализация метода performStepTimeSpec() для float.
    template<> 
    void NeursImplCuda<float>::performStepTimeSpec(
        const float & dt, 
        const std::valarray<float> & I,
        const float & VPeak,
        const float & VReset,
        const float & paramSpec,
        float & t,
        std::valarray<float> & V, 
        std::valarray<bool> & mask
    ) 
    {
        size_t n = I.size();
        size_t nSize = n*sizeof(float);
        size_t bSize = n*sizeof(bool);
        size_t size = sizeof(float);

        float *dtDev;
        float *tDev;
        float *paramSpecDev;
        float *VPeakDev;
        float *VResetDev;
        float *IDev;
        float *VDev;
        bool *maskDev;

        cudaMalloc( (void**)&dtDev, size);
        cudaMalloc( (void**)&tDev, size);
        cudaMalloc( (void**)&paramSpecDev, size);
        cudaMalloc( (void**)&VPeakDev, size);
        cudaMalloc( (void**)&VResetDev, size);
        cudaMalloc( (void**)&IDev, nSize);
        cudaMalloc( (void**)&VDev, nSize);
        cudaMalloc( (void**)&maskDev, bSize);

        cudaMemcpy( dtDev, &dt, size, cudaMemcpyHostToDevice );
        cudaMemcpy( tDev, &t, size, cudaMemcpyHostToDevice );
        cudaMemcpy( paramSpecDev, &paramSpec, size, cudaMemcpyHostToDevice );
        cudaMemcpy( VPeakDev, &VPeak, size, cudaMemcpyHostToDevice );
        cudaMemcpy( VResetDev, &VReset, size, cudaMemcpyHostToDevice );
        cudaMemcpy( IDev, &I[0], nSize, cudaMemcpyHostToDevice );
        cudaMemcpy( VDev, &V[0], nSize, cudaMemcpyHostToDevice );
        cudaMemcpy( maskDev, &mask[0], bSize, cudaMemcpyHostToDevice );

        performStepTimeSpecKernel<<< 1, n >>>( dtDev, IDev, VPeakDev, VResetDev, paramSpecDev, tDev, VDev, maskDev );

        cudaMemcpy( &t, tDev, size, cudaMemcpyDeviceToHost );
        cudaMemcpy( &V[0], VDev, nSize, cudaMemcpyDeviceToHost );
        cudaMemcpy( &mask[0], maskDev, bSize, cudaMemcpyDeviceToHost );

        cudaFree( dtDev );
        cudaFree( IDev );
        cudaFree( VPeakDev );
        cudaFree( VResetDev );
        cudaFree( paramSpecDev );
        cudaFree( tDev );
        cudaFree( VDev );
        cudaFree( maskDev );

    }

}

