# Declaration of variables
CC = g++
LFLAGS = -lpthread -xwindow

# File names
EXEC = hw2_nbody
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o) timer.o

# Main target
all: $(OBJECTS)
	$(CC) $(LFLAGS) $(OBJECTS) -o $(EXEC)

# To obtain object files
%.o: %.cpp
	$(CC) -c $< -o $@

timer.o: ../../timer.cpp
	$(CC) -c $< -o $@

# To clean all output files
clean:
	rm -f $(EXEC) $(OBJECTS)
