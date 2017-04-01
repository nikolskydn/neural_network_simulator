#include "solvercpu.hpp"
#include <memory>

namespace NNSimulator {

        //! Реализация тестового решателя на универсальных процессорах.
        template<class T>  void SolverImplCPU<T>::solveTest(
            const size_t & nN,
            const T & VP,
            const T & VR,
            const T &  dt,
            const T & st,
            const T & np,
            const T & cp,
            std::valarray<T> & V,
            std::valarray<bool> & m,
            std::valarray<T> & I,
            std::valarray<T> & w,
            T & t
        )
        {
            if( nN != ones_.size() )
            {
                ones_.resize( nN, static_cast<T>(1) );
                mInv_.resize( nN, false ); 
            }
            mInv_ = !m;
            while( t <= st ) 
            {
                V[mInv_] += np*dt*static_cast<std::valarray<T>>(I[mInv_]);
                V[m] = VR;
                m = V>VP;
                mInv_ = !m;
                I[m] += cp*dt*static_cast<std::valarray<T>>(V[m]);
                I[mInv_] *= (static_cast<T>(0.5)*ones_)[mInv_];
                t += dt;
            }
        }


        //! Интерфейс реализации модели Е.М. Ижикевича, принадлежащей классу сетей PCNN.
        template<class T>  void SolverImplCPU<T>::solvePCNN(
            const size_t & nN,
            const T & VP,
            const T & VR,
            const std::valarray<T> a,
            const std::valarray<T> b,
            const std::valarray<T> c,
            const std::valarray<T> d,
            const T &  dt,
            const T & st,
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
            while( t <= st ) 
            {
                //I=
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
