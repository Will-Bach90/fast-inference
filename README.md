# High-Performance ML Inference Runtime

A C++20 ML inference runtime focused on CPU performance, compiler-style graph optimization, and real-time video inference.

## Goals

* Implement a small tensor library from scratch
* Build a computation graph and execution runtime
* Implement optimized CPU kernels
* Explore SIMD, cache locality, memory reuse, and multithreading
* Implement graph optimizations such as operator fusion
* Build a memory planner
* Run small neural networks end-to-end
* Integrate the runtime into a real-time video pipeline
* Eventually add a CUDA backend

## Architecture
```
Model
  v
Computation Graph
  v
Graph Optimization
  ├── Operator Fusion
  └── Memory Planning
  v
Kernel Selection
  v
Execution Runtime
  ├── Scalar CPU
  ├── AVX2
  └── CUDA (planned)
  v
Video Pipeline
```

## Initial Operations

* Tensor creation and indexing
* Elementwise addition
* Multiplication
* ReLU
* Matrix multiplication
* Reductions
* Convolution

## Optimization Areas

* SIMD (AVX2/AVX-512)
* Cache-aware tiling
* Memory layout
* Buffer reuse
* Operator fusion
* Multithreading
* Thread affinity
* Memory allocation
* Kernel autotuning

## Benchmarking

Each optimization will be evaluated against a baseline using:

* Execution time
* Throughput
* Latency
* GFLOP/s
* Memory bandwidth
* CPU performance counters

The project will document both improvements and cases where an optimization does not help.

## Planned Stages

1. Tensor library
2. Scalar kernels
3. Benchmark and profiling infrastructure
4. SIMD kernels
5. Computation graph
6. Graph optimization
7. Memory planner
8. Neural-network inference
9. Real-time video pipeline
10. CUDA backend
11. GPU kernel optimization and autotuning

## Technology

* C++20
* CMake
* Linux
* OpenMP
* AVX2/AVX-512
* CUDA
* Python/NumPy/PyTorch for testing and reference implementations
