# GLIDE

This is the source code of GLIDE, which is our paper "GLIDE: GPU-Accelerated ANN Graph Index Construction via Data Locality".

## Directory description

- cmake: Third-party libraries required by GLIDE.
- datasets: The dataset required for GLIDE has the following format: the first 4 bytes (int32_t) indicates the size of the datasets, the second 4 bytes (int32_t) indicates the dimensions of the vectors, then each vector data (float) is closely packed. Note that for the Cosine dataset, the dataset needs to be normalized in advance.
- src: The source code for GLIDE.
- temp: Persistent data during GLIDE construction.

## Prerequisites

- CMake 3.26.4+
- GCC 9.3+ (11.4+ recommended)
- CUDA Toolkit 11.4+

## Build and run steps on Linux

Download this repository and change to the GLIDE folder.

```shell
mkdir build
cd build
cmake ..
make 
./test.sh
```