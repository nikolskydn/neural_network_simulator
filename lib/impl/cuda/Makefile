NVCC=nvcc
CPPFLAGS=--std=c++11 --gpu-architecture=compute_30
SRCS=$(wildcard *cu)
DLIB=$(SRCS:%.cu=libnn%.so)
DTESTLIB=$(SRCS:%.cu=libnn%test.so)

$(DLIB): libnn%.so: %.cu %.hpp
	$(NVCC) $(CPPFLAGS) --shared  -o $@ $< --compiler-options '-fPIC' 

$(DTESTLIB): libnn%test.so: %.cu %.hpp
	$(NVCC) $(CPPFLAGS) --shared  -o $@ $< --compiler-options '-fPIC' 
clean: 
	rm -f *.so
