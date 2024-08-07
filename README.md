# Tiled Algorithms for Flow-based Stencil Computation

This repository provides alternative tiled algorithms, **CfAMe** (memory-equivalent conflict-free tiled algorithm) and **CfAMo** (memory-optimized conflict-free tiled algorithm), to accelerate flow-based stencil computational models on Nvidia GPUs. 

The new algorithms are tested on the SciddicaT fluid-flow landslide simulator. The naive and classical tiled implementations, with and without halo cells, of SciddicaT are also provided for comparison.

The code can be compiled using the `Makefile` provided. To list the available targets, open a terminal window, go to the code directory, type `make`, a blank space, and press the TAB key twice.
To compile all the executables, use the following commands:

```
cd code
make
```

To run the naive implementation (on the standard Tessina simulation with 32x32 threads):

```
make run_cuda
```

To run CfAMe-based implementation (on the standard Tessina simulation with 32x32 threads):

```
make run_cuda_tiled_halo_sync
```

To run CfAMo-based implementation (on the standard Tessina simulation with 32x32 threads):

```
make run_cuda_tiled_halo_sync_priv
```

To run further experiments, use the run.sh and run_extended.sh:
```
bash run.sh -steps 4000 -gpu 0
```
or 
```
bash run_extended.sh -steps 4000 -gpu 0
```
## Authors

The code was developed by:
- **Donato D'Ambrosio**, **Salvatore Fiorentino**, **Bruno Francesco Barbara**, and **Agostino Rizzo**, Department of Mathematics and Computer Science, University of Calabria, Italy
- **Alessio De Rango** and **Giuseppe Mendicino**, Department of Environmental Engineering, University of Calabria, Italy
- **Pietro Sabatino**, Institute for High Performance Computing and Networking (ICAR-CNR), Italy

For information, please write an email to donato.dambrosio [at] unical.it

Please let us know if you use the code in this repository.

