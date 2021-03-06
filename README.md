# Parallel Programming Homeworks

This repository has all homeworks I have written in C++ in the parallel programming course of NTHU.

## Dependencies

Some programs require extra dependencies during linking. I concluded the dependencies those programs need in the following table:

| Homework Name | Dependencies |
| :------------- | :------------- |
| HW 2 - Roller Coaster Problem | pthread |
| HW 2 - N-body Problem - Pthread Implementation | pthread, X-Window |
| HW 2 - N-body Problem - OpenMP Implementation | X-Window |
| HW 2 - N-body Problem - BH-Algo Implementation | pthread, X-Window |
| HW 3 - OpenMP Implementation | X-Window |
| HW 3 - MPI, Hybrid Implementation | MPI, X-Window |

## Compile

There is a Makefile in the root directory. You can compile all source code by simply executing `make` or `make all`. All executable files will be generated in the root directory.

If you just want to compile the homeworks you need, please refer to the following table:

| Homework Name | Make Command |
| :------------- | :------------- |
| Entire Homework 2 | `make hw2` |
| HW 2 - Roller Coaster Problem | `make hw2_srcc` |
| HW 2 - N-body Problem - All Implementation | `make hw2_nbody` |
| HW 2 - N-body Problem - Pthread Implementation | `make hw2_nbody_pthread` |
| HW 2 - N-body Problem - OpenMP Implementation | `make hw2_nbody_omp` |
| HW 2 - N-body Problem - BH-Algo Implementation | `make hw2_nbody_gh` |
| Entire Homework 3 | `make hw3` |
| HW 3 - OpenMP - Static Scheduling Implementation | `make hw3_omp_static` |
| HW 3 - OpenMP - Dynamic Scheduling Implementation | `make hw3_omp_dynamic` |
| HW 3 - MPI - Static Scheduling Implementation | `make hw3_mpi_static` |
| HW 3 - MPI - Dynamic Scheduling Implementation | `make hw3_mpi_dynamic` |
| HW 3 - Hybrid - Static Scheduling Implementation | `make hw3_hybrid_static` |
| HW 3 - Hybrid - Dynamic Scheduling Implementation | `make hw3_hybrid_dynamic` |
