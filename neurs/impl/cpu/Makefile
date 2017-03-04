CXX=g++-4.9
CXXFLAGS=--std=c++14
SRCS=$(wildcard *cpp)
DLIB=$(SRCS:%.cpp=lib%.so)
SLIB=$(SRCS:%.cpp=lib%.a)

$(DLIB): lib%.so: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -fpic -shared $< -o $@

