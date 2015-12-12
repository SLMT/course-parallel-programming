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
OMP_MAIN_SOURCE = main.cpp main_openmp.cpp mandelbort_set.cpp
MPI_MAIN_SOURCE = main.cpp main_mpi.cpp mandelbort_set.cpp ms_mpi_static.cpp ms_mpi_dynamic.cpp

# Main target
all: omp_static omp_dynamic mpi_static mpi_dynamic hy_static hy_dynamic

omp_static: $(COMMON_OBJS)
	$(CC) $(LINK_FLAGS) $(COMMON_OBJS) -D OMP $(OMP_MAIN_SOURCE) -o $(OMP_STATIC_EXEC)

omp_dynamic: $(COMMON_OBJS)
	$(CC) $(LINK_FLAGS) $(COMMON_OBJS) -D OMP -D DYNAMIC $(OMP_MAIN_SOURCE) -o $(OMP_DYNAMIC_EXEC)

mpi_static: $(COMMON_OBJS)
	$(MPI_CC) $(LINK_FLAGS) $(COMMON_OBJS) $(MPI_MAIN_SOURCE) -o $(MPI_STATIC_EXEC)

mpi_dynamic: $(COMMON_OBJS)
	$(MPI_CC) $(LINK_FLAGS) $(COMMON_OBJS) -D DYNAMIC $(MPI_MAIN_SOURCE) -o $(MPI_DYNAMIC_EXEC)

hy_static: $(COMMON_OBJS)
	$(MPI_CC) $(LINK_FLAGS) $(COMMON_OBJS) -D HYBRID $(MPI_MAIN_SOURCE) -o $(HYBRID_STATIC_EXEC)

hy_dynamic: $(COMMON_OBJS)
	$(MPI_CC) $(LINK_FLAGS) $(COMMON_OBJS) -D HYBRID -D DYNAMIC $(MPI_MAIN_SOURCE) -o $(HYBRID_DYNAMIC_EXEC)

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
