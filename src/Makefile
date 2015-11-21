# Declaration of variables
CC = g++
BUILD_DIR = ..

# Main target
all: $(OBJECTS)

## Homework 2
hw2: hw2_srcc

# hw2 - srcc
HW2_SRCC_DIR = 
HW2_SRCC_SOURCES = pp/hw2/

hw2_srcc:
	$(MAKE) -C pp/hw2/rccsim
	cd pp && $(CC) -c timer.cpp -o ../$(BUILD_DIR)/timer.o
	$(CC)  -lpthread timer.cpp
	$(CC) $(LINK_FLAGS) $(OBJECTS) -o $(EXEC)
