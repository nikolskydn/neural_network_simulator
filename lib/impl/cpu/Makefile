CXX=g++
CXXFLAGS=-std=c++14 -O3 -fopenmp
SRCS=$(wildcard *.cpp)
DLIBS=$(SRCS:%.cpp=libnn%.so)
DTESTLIBS=$(SRCS:%.cpp=libnn%test.so)

$(DLIBS): libnn%.so: %.cpp %.hpp ../../setting.h
	$(CXX) $(CXXFLAGS) -fpic -shared $< -o $@

$(DTESTLIBS): libnn%test.so: %.cpp %.hpp ../../setting.h
	$(CXX) $(CXXFLAGS) -fpic -shared $< -o $@
clean:
	rm -f *.so
