CXX=g++
CXXFLAGS=--std=c++14
SRCS=$(wildcard *.cpp)
DLIBS=$(SRCS:%.cpp=libnn%.so)

all: $(DLIBS)

$(DLIBS): libnn%.so: %.cpp %.hpp ../../setting.h
	$(CXX) $(CXXFLAGS) -fpic -shared $< -o $@
clean:
	rm -f *.so

