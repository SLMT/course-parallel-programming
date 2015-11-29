# Declaration of variables
CC = g++

# File names
PTHREAD_EXEC = hw2_NB_pthread
OMP_EXEC = hw2_NB_omp
GH_EXEC = hw2_NB_BHalgo
COMMON_OBJS = nbody.o gui.o bh_tree.o

# Main target
all: pthread omp gh

pthread: $(COMMON_OBJS)
	$(CC) -c -o nbody_pthread.o nbody_pthread.cpp
	$(CC) -c main.cpp -o main.o
	$(CC) -lpthread -lX11 $(COMMON_OBJS) nbody_pthread.o main.o -o $(PTHREAD_EXEC)

omp: $(COMMON_OBJS)
	$(CC) -c -fopenmp nbody_omp.cpp -o nbody_omp.o
	$(CC) -c -D OMP main.cpp -o main.o
	$(CC) -lpthread -lX11 -fopenmp $(COMMON_OBJS) nbody_omp.o main.o -o $(OMP_EXEC)

gh: $(COMMON_OBJS)
	$(CC) -c -D BH_ALGO -o nbody_pthread.o nbody_pthread.cpp
	$(CC) -c main.cpp -o main.o
	$(CC) -lpthread -lX11 $(COMMON_OBJS) nbody_pthread.o main.o -o $(GH_EXEC)

# To obtain object files
%.o: %.cpp
	$(CC) -c $< -o $@

# To clean all output files
clean:
	rm -f *.o $(PTHREAD_EXEC) $(OMP_EXEC) $(GH_EXEC)
