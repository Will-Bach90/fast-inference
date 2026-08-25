#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include "tensor.hpp"
#include <arm_neon.h>

namespace ops {
    void add(const Tensor& a, const Tensor& b, Tensor& output);
    void multiply(const Tensor& a, const Tensor& b, Tensor& output);
    void relu(const Tensor& input, Tensor& output);

    void matmul_naive(const Tensor& a, const Tensor& b, Tensor& output);
    void matmul_ikj(const Tensor& a, const Tensor& b, Tensor& output);
    void matmul_tiled(const Tensor& a, const Tensor& b, Tensor& output, const size_t bs);
    void matmul_neon(const Tensor& a, const Tensor& b, Tensor& output);
    void matmul_neon_ikj(const Tensor& a, const Tensor& b, Tensor& output);
    void matmul_4x4(const Tensor& a, const Tensor& b, Tensor& output);
    void matmul_8x4(const Tensor& a, const Tensor& b, Tensor& output);
    void matmul_12x4(const Tensor& a, const Tensor& b, Tensor& output);
    void matmul_14x4(const Tensor& a, const Tensor& b, Tensor& output);
    void matmul_16x4(const Tensor& a, const Tensor& b, Tensor& output);

    static inline float32x4_t matmul_kernel(const float* a, const float* b, float32x4_t c, size_t K, size_t N);
    static inline void matmul_kernel_4x4(const float* a, const float* b, float* c, size_t K, size_t N);
    static inline void matmul_kernel_8x4(const float* a, const float* b, float* c, size_t K, size_t N);
    static inline void matmul_kernel_12x4(const float* a, const float* b, float* c, size_t K, size_t N);
    static inline void matmul_kernel_14x4(const float* a, const float* b, float* c, size_t K, size_t N);
    static inline void matmul_kernel_16x4(const float* a, const float* b, float* c, size_t K, size_t N);
}

#endif
