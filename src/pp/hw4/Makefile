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

COMMON_OBJS = io.o main.o timer.o

# Main target
all: single_cuda openmp_cuda mpi_cuda

seq_apsp: $(COMMON_OBJS) apsp_seq.o
	$(CC) $(COMMON_OBJS) apsp_seq.o -o $(SEQ_EXEC)

single_cuda: $(COMMON_OBJS) apsp_single_gpu.o block_calculation.o
	$(CUDA_CC) $(COMMON_OBJS) apsp_single_gpu.o block_calculation.o -o $(SINGLE_CUDA_EXEC)

# To obtain object files
%.o: %.cpp
	$(CC) -c $< -o $@

timer.o: ../timer.cpp
	$(CC) -c $< -o $@

block_calculation.o: block_calculation.cu
	$(CUDA_CC) -c $< -o $@

apsp_single_gpu.o: apsp_single_gpu.cu
	$(CUDA_CC) -c $< -o $@

# To clean all output files
clean:
	rm -f *.o $(SINGLE_CUDA_EXEC) $(OPENMP_CUDA_EXEC) $(MPI_CUDA_EXEC)
