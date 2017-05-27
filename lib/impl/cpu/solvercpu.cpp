#include "solvercpu.hpp"
#include <memory>
#include <chrono>
#include <cmath>
#include <omp.h>
#include "../../setting.h"

#include <iostream>


namespace NNSimulator {

    //template<class T> void SolverImplCPU<T>::SolverImplCPU()
    //{
    //}


        //! Реализация модели Е.М. Ижикевича (2003).
        template<class T>  void SolverImplCPU<T>::solvePCNNI2003E(
            const size_t & nN,
            const size_t & nE,
            const T & VP,
            const std::valarray<T> & a,
            const std::valarray<T> & b,
            const std::valarray<T> & c,
            const std::valarray<T> & d,
            const std::valarray<T> & w,
            const T &  dt,
            const T & te,
            std::valarray<T> & V,
            std::valarray<T> & U,
            std::valarray<bool> & m,
            std::valarray<T> & I,
            T & t,
            std::deque<std::pair<T,std::valarray<T>>> & og
        ) 
        {

            #ifdef TimeDebug 
                auto bTime = std::chrono::steady_clock::now();
                auto sst = t;
            #endif

            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> dis;

            T dt_2 = .5*dt;
            T k1 = .04;
            T k2 = 5.;
            T k3 = 140.;
            T i1 = 5.;
            T i2 = 2.;
            T one = 1.;

            #ifdef NN_TEST_SOLVERS
                std::cout << "\033[31;1m+++ warning: test mode\033[0m\n";
            #endif

            size_t i;
            while( t < te ) 
            {
                #pragma omp parallel for
                for( i=0; i<nN; ++i)
                {
                    if( i<nE ) I[i] = i1;
                    else I[i] = i2;
                    #ifndef NN_TEST_SOLVERS
                        I[i] *= dis(gen);
                    #endif
                    if( m[i] )
                    {
                        V[i] = c[i];
                        U[i] += d[i];
                    }
                    std::valarray<T> row = w[std::slice(i*nN,nN,1)];
                    std::valarray<T> rowm = row[m];
                    I[i] += rowm.sum();
                    V[i] += dt_2*( k1*V[i]*V[i] + k2*V[i] + k3 - U[i] + I[i] );
                    V[i] += dt_2*( k1*V[i]*V[i] + k2*V[i] + k3 - U[i] + I[i] );
                    U[i] += dt*a[i]*( b[i] * V[i] - U[i] );
                }
                #pragma omp parallel for
                for( size_t i=0; i<nN; ++i)
                {
                    m[i] = V[i]>=VP;
                }
                t += dt;
                og.push_back(std::pair<T,std::valarray<T>>(t,V));
            }
            #ifdef TimeDebug 
                auto eTime = std::chrono::steady_clock::now();
                auto dTime = std::chrono::duration_cast<std::chrono::milliseconds>(eTime-bTime);
                std::cout << "Neurons: " << nN << std::endl;
                std::cout << "Connects: " << nN*nN << std::endl;
                std::cout << "Steps: " << std::ceil((te-sst)/dt) << std::endl;
                std::cout << "solvePCNNI2003E_CPU time " << dTime.count() << ", ms\n";
            #endif

        }

    template class SolverImplCPU<float>; 

    template class SolverImplCPU<double>; 
}
