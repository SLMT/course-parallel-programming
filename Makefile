### Declare variables
BIN_DIR = .

### Source path
HW2_SRCC_DIR = src/pp/hw2/rccsim
HW2_NBODY_DIR = src/pp/hw2/nbody

## HW2
hw2: hw2_srcc hw2_nbody

hw2_srcc:
	cd $(HW2_SRCC_DIR) && $(MAKE) all
	mv $(HW2_SRCC_DIR)/hw2_SRCC $(BIN_DIR)
	cd $(HW2_SRCC_DIR) && $(MAKE) clean

hw2_nbody: hw2_nbody_pthread

hw2_nbody_pthread:
	cd $(HW2_NBODY_DIR) && $(MAKE) pthread
	mv $(HW2_NBODY_DIR)/hw2_NB_pthread $(BIN_DIR)
	cd $(HW2_NBODY_DIR) && $(MAKE) clean
