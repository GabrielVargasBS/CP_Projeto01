# Compila a versão sequencial por padrao
VERSION = Sequencial
CPP_COMPILER = mpic++ -std=c++11 -Wall -g
CUDA_COMPILER = nvcc -std=c++11 -g
SRC_DIR = srcSequencial

ifeq ($(VERSION),OpenMP)
	SRC_DIR = srcOpenMP
	CPP_COMPILER = mpic++ -std=c++11 -Wall -g -DNUM_THREADS=$(NUM_THREADS) -fopenmp
else ifeq ($(VERSION),MPI)
	SRC_DIR = srcMPI
	CPP_COMPILER = mpic++ -std=c++11 -Wall -g -DNUM_THREADS=$(NUM_THREADS) -fopenmp
else ifeq ($(VERSION),OpenMP_GPU)
	SRC_DIR = srcOpenMP_GPU
	CPP_COMPILER = mpic++ -std=c++11 -Wall -g -DNUM_THREADS=$(NUM_THREADS) -fopenmp
else ifeq ($(VERSION),CUDA)
	SRC_DIR = srcCUDA
else
	SRC_DIR = srcSequencial
	CPP_COMPILER = mpic++ -std=c++11 -Wall -g
endif

EXEC_PROG = neuralnetwork
BINARIES = $(EXEC_PROG)

CPP_SOURCES := $(shell find $(SRC_DIR) -name '*.cpp')
CU_SOURCES := $(shell find $(SRC_DIR) -name '*.cu')

CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)

OBJECTS = main.o $(CPP_OBJECTS) $(CU_OBJECTS)

all: clean $(EXEC_PROG)
	@echo Neural Network Build Completed

%.o: %.cpp
	$(CPP_COMPILER) -c -o $@ $< -w

%.o: %.cu
	$(CUDA_COMPILER) -c -o $@ $< -w

$(EXEC_PROG): $(OBJECTS)
	$(CPP_COMPILER) -o $(EXEC_PROG) $(OBJECTS) 

.PHONY : run
run:
	./$(EXEC_PROG)

.PHONY : clean 
clean:
	rm -rf $(EXEC_PROG) $(shell find . -name '*.o')