# Declaration of variables
CC = g++

# File names
PTHREAD_EXEC = hw2_NB_pthread
OMP_EXEC = hw2_NB_omp
COMMON_OBJS = nbody.o

# Main target
all: pthread omp

pthread: $(COMMON_OBJS) nbody_pthread.o
	$(CC) -c main.cpp -o main.o
	$(CC) -lpthread $(COMMON_OBJS) nbody_pthread.o main.o -o $(PTHREAD_EXEC)

omp: $(COMMON_OBJS)
	$(CC) -c -fopenmp nbody_omp.cpp -o nbody_omp.o
	$(CC) -c -D OMP main.cpp -o main.o
	$(CC) -lpthread -fopenmp $(COMMON_OBJS) nbody_omp.o main.o -o $(OMP_EXEC)

# To obtain object files
%.o: %.cpp
	$(CC) -c $< -o $@

# To clean all output files
clean:
	rm -f *.o $(PTHREAD_EXEC) $(OMP_EXEC)
