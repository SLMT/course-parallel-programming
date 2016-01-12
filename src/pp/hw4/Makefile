# Declaration of variables
CC = g++
MPI_CC = mpic++
CUDA_CC = nvcc

OMP_FLAG = -Xcompiler="-fopenmp"

MPI_COMPILE_FLAG = -I/usr/include/mpich-x86_64
MPI_LINK_FLAG = -L/usr/lib64/mpich/lib -lmpich

# File names
SEQ_EXEC = seq
SINGLE_GPU_EXEC = hw4_cuda
OPENMP_GPU_EXEC = hw4_openmp
MPI_GPU_EXEC = hw4_mpi

COMMON_OBJS = io.o timer.o

# Main target
all: single omp mpi

seq_apsp: normal_main $(COMMON_OBJS) apsp_seq.o
	$(CC) $(COMMON_OBJS) main.o apsp_seq.o -o $(SEQ_EXEC)

single: normal_main $(COMMON_OBJS) apsp_single_gpu.o block_calculation.o
	$(CUDA_CC) $(COMMON_OBJS) main.o apsp_single_gpu.o block_calculation.o -o $(SINGLE_GPU_EXEC)

omp: normal_main $(COMMON_OBJS) apsp_omp.o block_calculation.o
	$(CUDA_CC) $(OMP_FLAG) $(COMMON_OBJS) main.o apsp_omp.o block_calculation.o -o $(OPENMP_GPU_EXEC)

mpi: mpi_main $(COMMON_OBJS) apsp_mpi.o block_calculation.o
	$(CUDA_CC) $(MPI_LINK_FLAG) $(COMMON_OBJS) main.o apsp_multi_gpu.o block_calculation.o -o $(MPI_GPU_EXEC)

# Different types of main file
normal_main: main.cpp
	$(CC) -c -o main.o main.cpp

mpi_main: main.cu
	$(CUDA_CC) $(MPI_COMPILE_FLAG) -c -D MPI_IMPL -o main.o main.cu

main.cu: main.cpp
	cp main.cpp main.cu

# To obtain object files
%.o: %.cpp
	$(CC) -c $< -o $@

timer.o: ../timer.cpp
	$(CC) -c $< -o $@

block_calculation.o: block_calculation.cu
	$(CUDA_CC) -c $< -o $@

apsp_single_gpu.o: apsp_single_gpu.cu
	$(CUDA_CC) -c $< -o $@

apsp_omp.o: apsp_omp.cu
	$(CUDA_CC) $(OMP_FLAG) -c $< -o $@

apsp_mpi.o: apsp_mpi.cu
	$(CUDA_CC) $(MPI_COMPILE_FLAG) -c $< -o $@

# To clean all output files
clean:
	rm -f *.o $(SINGLE_GPU_EXEC) $(OPENMP_GPU_EXEC) $(MPI_GPU_EXEC)
