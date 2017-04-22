#include "solvercpu.hpp"
#include <memory>
#include <chrono>
#include <cmath>
#include "../../setting.h"

//#ifdef NN_TEST_SOLVERS
    #include <iostream>
//#endif

#define CPUValArrayTimeDebug 1

namespace NNSimulator {

    //! Заполняет входной массив случайными числами из нормалного распределения с использованием ГПСЧ "Вихрь Мерсенна".
    template<class T> void SolverImplCPU<T>::makeRandn( std::valarray<T> & v )
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis;
        for(auto & e: v) e = dis(gen);
    }

        //! Реализация модели Е.М. Ижикевича (2003).
        template<class T>  void SolverImplCPU<T>::solvePCNNI2003E(
            const size_t & nN,
            const size_t & nE,
            const T & VP,
            const T & VR,
            const std::valarray<T> a,
            const std::valarray<T> b,
            const std::valarray<T> c,
            const std::valarray<T> d,
            const T &  dt,
            const T & te,
            std::valarray<T> & V,
            std::valarray<T> & U,
            std::valarray<bool> & m,
            std::valarray<T> & I,
            std::valarray<T> & w,
            T & t,
            std::vector<std::pair<size_t,T>> & sp,
            std::vector<std::pair<size_t,std::valarray<T>>> & og
        ) 
        {
#if  CPUValArrayTimeDebug > 0
            auto bTime = std::chrono::steady_clock::now();
#endif

            T dt_2 = .5*dt;
            T k1 = .04;
            T k2 = 5.;
            T k3 = 140.;
            T i1 = 5.;
            T i2 = 2.;
            std::valarray<T> rV(nN);
            while( t < te ) 
            {
                #ifndef NN_TEST_SOLVERS
                    makeRandn(rV);
                    I[std::slice(0,nE,1)] = i1;
                    I[std::slice(nE,nN-nE,1)] = i2;
                    I *= rV;
                #else
                    std::cout << "\033[31;1m--- warning: test mode\033[0m\n";
                #endif
                V[m] = c[m];
                U[m] += d[m];
                for( size_t i=0; i<nN; ++i)
                {
                    std::valarray<T> row = w[std::slice(i*nN,nN,1)];
                    std::valarray<T> rowm = row[m];
                    I[i] += rowm.sum();
                }
                V += dt_2*( k1*V*V + k2*V + k3 - U + I );
                V += dt_2*( k1*V*V + k2*V + k3 - U + I );
                U += dt*a*( b * V - U );
                t += dt;
                m = V>=VP;

                for( size_t i=0; i<nN; ++i )
                {
                    if( m[i] ) sp.push_back(std::pair<T,size_t>(t,i));
                }
                og.push_back(std::pair<T,std::valarray<T>>(t,V));
            }
#if  CPUValArrayTimeDebug > 0
                auto eTime = std::chrono::steady_clock::now();
                auto dTime = std::chrono::duration_cast<std::chrono::milliseconds>(eTime-bTime);
                std::cout << "Neurons: " << nN << std::endl;
                std::cout << "Connects: " << nN*nN << std::endl;
                std::cout << "Steps: " << std::ceil(te/dt) << std::endl;
                std::cout << "solvePCNNI2003E_CPU time " << dTime.count() << ", ms\n";
#endif

        }

    template class SolverImplCPU<float>; 

    template class SolverImplCPU<double>; 
}
