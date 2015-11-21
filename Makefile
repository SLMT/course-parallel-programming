### Declare variables
BIN_DIR = .

### Source path
HW2_SRCC_DIR = src/pp/hw2/rccsim

## HW2
hw2: hw2_srcc

hw2_srcc:
	cd $(HW2_SRCC_DIR) && $(MAKE) all
	mv $(HW2_SRCC_DIR)/hw2_SRCC $(BIN_DIR)
	cd $(HW2_SRCC_DIR) && $(MAKE) clean
