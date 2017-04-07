NVCC=nvcc
CPPFLAGS=--std=c++11
SRCS=$(wildcard *cu)
DLIB=$(SRCS:%.cu=libnn%.so)
#SLIB=$(SRCS:%.cu=lib%.a)

$(DLIB): libnn%.so: %.cu %.hpp
	$(NVCC) $(CPPFLAGS) --shared  -o $@ $< --compiler-options '-fPIC' 

#$(SLIB): lib%.a: %.cu %.hpp
#	$(NVCC) $(CPPFLAGS) --lib --output-file $@ $<

clean: 
	rm -f *.so
#	rm -f *.a
