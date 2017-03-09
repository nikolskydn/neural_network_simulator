CXX=g++-4.9
CXXFLAGS=--std=c++14
SRCS=$(wildcard *.cpp)
DLIBS=$(SRCS:%.cpp=lib%.so)

all: $(DLIBS)

$(DLIBS): lib%.so: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -fpic -shared $< -o $@
clean:
	rm -f *.so

