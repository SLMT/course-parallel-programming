# Declaration of variables
CC = g++

# File names
PTHREAD_EXEC = hw2_NB_pthread
PTHREAD_OBJS = main.o nbody_pthread.o nbody.o

# Main target
all: pthread

pthread: $(PTHREAD_OBJS)
	$(CC) -lpthread $(PTHREAD_OBJS) -o $(PTHREAD_EXEC)

# To obtain object files
%.o: %.cpp
	$(CC) -c $< -o $@

# To clean all output files
clean:
	rm -f $(PTHREAD_OBJS) $(PTHREAD_EXEC)
