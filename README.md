The project is designed to create simulators of neural and neuroglial networks. 
The development is realized in C ++. The project architecture is presented in the diagram.

The abstract class Solver defines the interface of simulators for neural and neuroglial networks. 
The creation of specific simulators, representing descendants of the Solver class, 
is performed in accordance with the Factory method pattern. 
The latter allows you to expand the project with your own models.

The implementation of descendant's  methods  of Solver classes is separated 
from their interface in accordance with the Bridge pattern. 
The latter allows you to build the project for universal and graphics processors. 
Implementations for graphics processors are made using Cuda technology.

@image html doc/img/schemesolver.png
