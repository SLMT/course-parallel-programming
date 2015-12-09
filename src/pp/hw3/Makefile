# Declaration of variables
CC = g++
MPI_CC = mpi++

# File names
OMP_STATIC_EXEC = MS_OpenMP_static
OMP_DYNAMIC_EXEC = MS_OpenMP_dynamic
MPI_STATIC_EXEC = MS_MPI_static
MPI_DYNAMIC_EXEC = MS_MPI_dynamic
HYBRID_STATIC_EXEC = MS_Hybrid_static
HYBRID_DYNAMIC_EXEC = MS_Hybrid_dynamic
COMMON_OBJS = mandelbort_set.o gui.o

# Main target
all: omp_static omp_dynamic mpi_static

omp_static: $(COMMON_OBJS)
	$(CC) -c -fopenmp mandelbort_set_openmp.cpp -o mandelbort_set_openmp.o
	$(CC) -c -D OMP main.cpp -o main.o
	$(CC) -lX11 -fopenmp $(COMMON_OBJS) mandelbort_set_openmp.o main.o -o $(OMP_STATIC_EXEC)

omp_dynamic: $(COMMON_OBJS)
	$(CC) -c -D DYNAMIC -fopenmp mandelbort_set_openmp.cpp -o mandelbort_set_openmp.o
	$(CC) -c -D OMP main.cpp -o main.o
	$(CC) -lX11 -fopenmp $(COMMON_OBJS) mandelbort_set_openmp.o main.o -o $(OMP_DYNAMIC_EXEC)

mpi_static: $(COMMON_OBJS)
	$(MPI_CC) -c mandelbort_set_mpi.cpp -o mandelbort_set_mpi.o
	$(MPI_CC) -c main.cpp -o main.o
	$(MPI_CC) -lX11 $(COMMON_OBJS) mandelbort_set_mpi.o main.o -o $(MPI_STATIC_EXEC)

# To obtain object files
%.o: %.cpp
	$(CC) -c $< -o $@

gui.o: ../gui.cpp
	$(CC) -c $< -o $@

timer.o: ../timer.cpp
	$(CC) -c $< -o $@

# To clean all output files
clean:
	rm -f *.o $(COMMON_OBJS) $(OMP_STATIC_EXEC) $(OMP_DYNAMIC_EXEC) $(MPI_STATIC_EXEC) $(MPI_DYNAMIC_EXEC) $(HYBRID_STATIC_EXEC) $(HYBRID_DYNAMIC_EXEC)
