#include "solvercpu.hpp"
#include <memory>

namespace NNSimulator {

        //! Реализация решателя на универсальных процессорах.
        template<class T>  void SolverImplCPU<T>::solveExplicitEuler(
            std::unique_ptr<NNSimulator::Neurs<T>> & neurs,
            std::unique_ptr<NNSimulator::Conns<T>> & conns,
            const T & dt,
            const T & simulationTime 
        )
        {
            neurs->performStepTimeInit();
            conns->performStepTimeInit();
            T time = neurs->getTime();
            while( time < simulationTime )
            {
                neurs->performStepTime(dt);
                conns->performStepTime(dt);
                time += dt;
            }
            neurs->performStepTimeFinalize();
            conns->performStepTimeFinalize();
        }

    template class SolverImplCPU<float>; 

    template class SolverImplCPU<double>; 
}
