### Declare variables
BIN_DIR = .

### Source path
HW2_SRCC_DIR = src/pp/hw2/rccsim
HW2_NBODY_DIR = src/pp/hw2/nbody
HW3_DIR = src/pp/hw3

## All
all: hw2 hw3

## HW2
hw2: hw2_srcc hw2_nbody

hw2_srcc:
	cd $(HW2_SRCC_DIR) && $(MAKE) all
	mv $(HW2_SRCC_DIR)/hw2_SRCC $(BIN_DIR)
	cd $(HW2_SRCC_DIR) && $(MAKE) clean

hw2_nbody: hw2_nbody_pthread hw2_nbody_omp hw2_nbody_gh

hw2_nbody_pthread:
	cd $(HW2_NBODY_DIR) && $(MAKE) pthread
	mv $(HW2_NBODY_DIR)/hw2_NB_pthread $(BIN_DIR)
	cd $(HW2_NBODY_DIR) && $(MAKE) clean

hw2_nbody_omp:
	cd $(HW2_NBODY_DIR) && $(MAKE) omp
	mv $(HW2_NBODY_DIR)/hw2_NB_omp $(BIN_DIR)
	cd $(HW2_NBODY_DIR) && $(MAKE) clean

hw2_nbody_gh:
	cd $(HW2_NBODY_DIR) && $(MAKE) gh
	mv $(HW2_NBODY_DIR)/hw2_NB_BHalgo $(BIN_DIR)
	cd $(HW2_NBODY_DIR) && $(MAKE) clean

## hw3
hw3: hw3_omp_static hw3_omp_dynamic hw3_mpi_static hw3_mpi_dynamic hw3_hybrid_static hw3_hybrid_dynamic

hw3_omp_static:
	cd $(HW3_DIR) && $(MAKE) omp_static
	mv $(HW3_DIR)/MS_OpenMP_static $(BIN_DIR)
	cd $(HW3_DIR) && $(MAKE) clean

hw3_omp_dynamic:
	cd $(HW3_DIR) && $(MAKE) omp_dynamic
	mv $(HW3_DIR)/MS_OpenMP_dynamic $(BIN_DIR)
	cd $(HW3_DIR) && $(MAKE) clean

hw3_mpi_static:
	cd $(HW3_DIR) && $(MAKE) mpi_static
	mv $(HW3_DIR)/MS_MPI_static $(BIN_DIR)
	cd $(HW3_DIR) && $(MAKE) clean

hw3_mpi_dynamic:
	cd $(HW3_DIR) && $(MAKE) mpi_dynamic
	mv $(HW3_DIR)/MS_MPI_dynamic $(BIN_DIR)
	cd $(HW3_DIR) && $(MAKE) clean

hw3_hybrid_static:
	cd $(HW3_DIR) && $(MAKE) hy_static
	mv $(HW3_DIR)/MS_Hybrid_static $(BIN_DIR)
	cd $(HW3_DIR) && $(MAKE) clean

hw3_hybrid_dynamic:
	cd $(HW3_DIR) && $(MAKE) hy_dynamic
	mv $(HW3_DIR)/MS_Hybrid_dynamic $(BIN_DIR)
	cd $(HW3_DIR) && $(MAKE) clean
