# Declaration of variables
CC = g++

# File names
COMMON_OBJS = mandelbort_set.o

# Main target
all: test

test: $(COMMON_OBJS)
	$(CC) $(COMMON_OBJS) -o testexe

# To obtain object files
%.o: %.cpp
	$(CC) -c $< -o $@

timer.o: ../timer.cpp
	$(CC) -c $< -o $@

# To clean all output files
clean:
	rm -f *.o $(PTHREAD_EXEC) $(OMP_EXEC) $(GH_EXEC)
