# Declaration of variables
CC = g++
MPI_CC = mpic++
CUDA_CC = nvcc
LINK_FLAGS = -lX11 -fopenmp

# File names
SEQ_EXEC = seq
SINGLE_CUDA_EXEC = single_cuda
OPENMP_CUDA_EXEC = openmp_cuda
MPI_CUDA_EXEC = mpi_cuda

COMMON_SOURCE = io.cpp main.cpp ../timer.cpp

# Main target
all: single_cuda openmp_cuda mpi_cuda

seq_apsp:
	$(CC) $(COMMON_SOURCE) apsp_seq.cpp -o $(SEQ_EXEC)

single_cuda:
	$(CUDA_CC) $(COMMON_SOURCE) apsp_single_cuda.cu -o $(SINGLE_CUDA_EXEC)

# To clean all output files
clean:
	rm -f *.o $(SINGLE_CUDA_EXEC) $(OPENMP_CUDA_EXEC) $(MPI_CUDA_EXEC)
