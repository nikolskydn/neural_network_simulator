#include "solvercpu.hpp"
#include <memory>
#include "../../setting.h"

#ifdef NN_TEST_SOLVERS
    #include <iostream>
#endif

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
        template<class T>  void SolverImplCPU<T>::solvePCNN2003E(
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
            T & t
        ) 
        {
            T dt_2 = .5*dt;
            T k1 = .04;
            T k2 = 5.;
            T k3 = 140.;
            //T p = 2.;
            while( t < te ) 
            {
                #ifndef NN_TEST_SOLVERS
                    makeRandn(I);
                #else
                    std::cout << "\033[31;1m--- warning: test mode\033[0m\n";
                #endif
                m = V>=VP;
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
            }
        }


    template class SolverImplCPU<float>; 

    template class SolverImplCPU<double>; 
}
