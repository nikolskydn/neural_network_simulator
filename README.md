The project is designed to create simulators of neural and neuroglial networks. 
The development is realized in C++. The project architecture is presented in the diagram.

The abstract class Solver defines the interface of simulators for neural and neuroglial networks. 
The creation of specific simulators, representing descendants of the Solver class, 
is performed in accordance with the Factory method pattern. 
The latter allows you to expand the project with your own models.

The implementation of descendant's  methods  of Solver classes is separated 
from their interface in accordance with the Bridge pattern. 
The latter allows you to build the project for universal and graphics processors. 
Implementations for graphics processors are made using Cuda technology.

![Alt text](https://raw.githubusercontent.com/nikolskydn/neural_network_simulator/master/doc/img/schemesolver.png "Diagram")

The Date class and its heirs contain the network settings.

![Alt text](https://github.com/nikolskydn/neural_network_simulator/blob/master/doc/img/schemedata.png)

# Compiling

In order to compile the simulator use makerCPU.sh bash-script. If you have NVIDIA CUDA, you can also use makerCuda.sh script for compiling Izhikevich simulator which uses your video card.

Example: (in git folder)
```
mkdir compile
cd ./simulator/
./makerCPU.sh ../compile/ ../compile/
./makerCuda.sh ../compile ../compile/
```

In order to compile BOOST tests use makercpu.sh and makercuda.sh bash-scripts in ./tests/ folder.

Example: (in git folder)
```
mkdir -p compile/tests
cd ./tests/
./makercpu.sh ../compile/tests/ ../compile/tests/
./makercuda.sh ../compile/tests/ ../compile/tests/
```
