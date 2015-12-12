# Declaration of variables
CC = g++
MPI_CC = mpic++
LINK_FLAGS = -lX11 -fopenmp

# File names
OMP_STATIC_EXEC = MS_OpenMP_static
OMP_DYNAMIC_EXEC = MS_OpenMP_dynamic
MPI_STATIC_EXEC = MS_MPI_static
MPI_DYNAMIC_EXEC = MS_MPI_dynamic
HYBRID_STATIC_EXEC = MS_Hybrid_static
HYBRID_DYNAMIC_EXEC = MS_Hybrid_dynamic

COMMON_OBJS = gui.o timer.o
MPI_VER_OBJS = mandelbort_set.o ms_mpi_static.o ms_mpi_dynamic.o

# Main target
all: omp_static omp_dynamic mpi_static mpi_dynamic hy_static hy_dynamic

omp_static: $(COMMON_OBJS) main_openmp.o
	$(CC) -c -fopenmp mandelbort_set.cpp -o mandelbort_set.o
	$(CC) -c -D OMP main.cpp -o main.o
	$(CC) $(LINK_FLAGS) $(COMMON_OBJS) mandelbort_set.o main_openmp.o main.o -o $(OMP_STATIC_EXEC)

omp_dynamic: $(COMMON_OBJS) main_openmp.o
	$(CC) -c -D DYNAMIC -fopenmp mandelbort_set.cpp -o mandelbort_set.o
	$(CC) -c -D OMP main.cpp -o main.o
	$(CC) $(LINK_FLAGS) $(COMMON_OBJS) mandelbort_set.o main_openmp.o main.o -o $(OMP_DYNAMIC_EXEC)

mpi_static: $(COMMON_OBJS) mpi_objects
	$(MPI_CC) -c main_mpi.cpp -o main_mpi.o
	$(MPI_CC) -c main.cpp -o main.o
	$(MPI_CC) $(LINK_FLAGS) $(COMMON_OBJS) $(MPI_VER_OBJS) main_mpi.o main.o -o $(MPI_STATIC_EXEC)

mpi_dynamic: $(COMMON_OBJS) mpi_objects
	$(MPI_CC) -c -D DYNAMIC main_mpi.cpp -o main_mpi.o
	$(MPI_CC) -c main.cpp -o main.o
	$(MPI_CC) $(LINK_FLAGS) $(COMMON_OBJS) $(MPI_VER_OBJS) main_mpi.o main.o -o $(MPI_DYNAMIC_EXEC)

hy_static: $(COMMON_OBJS)

hy_dynamic: $(COMMON_OBJS)

# Compile Sets
mpi_objects:
	$(CC) -c -fopenmp mandelbort_set.cpp -o mandelbort_set.o
	$(MPI_CC) -c ms_mpi_static.cpp -o ms_mpi_static.o
	$(MPI_CC) -c ms_mpi_dynamic.cpp -o ms_mpi_dynamic.o

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
