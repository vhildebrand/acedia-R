# Makefile for compiling R/CUDA wrapper


R_HOME = $(shell R RHOME)

# R headers and compile position-independent code
CPPFLAGS = -I"$(R_HOME)/include" -D_FORTIFY_SOURCE=2 -fPIC


# Linker flags to link against R's librarires
LDFLAGS = -L"$(R_HOME)/lib" -lR


# CUDA comiper and flags
NVCC = nvcc
NVCCFLAGS = -03 -arch=sm_89 --ptxas-options=-v -c -fPIC


# name of final shared library
TARGET_LIB = gpu_add.so

# source files
R_SRC = R_interface.cpp
CUDA_SRC = vector_add.cu
# object files
R_OBJ = $(R_SRC:.cpp=.o)
CUDA_OBJ = $(CUDA_SRC:.cu=.o)


# default target
all: $(TARGET_LIB)

$(TARGET_LIB): $(R_OBJ) $(CUDA_OBJ)
	g++ -shared -o $@ $^ $(LDFLAGS)

$(R_OBJ): $(R_SRC)
	g++ $(CPPFLAGS) -c $< -o $@


$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@


clean:
	rm -f $(TARGET_LIB) $(R_OBJ) $(CUDA_OBJ)