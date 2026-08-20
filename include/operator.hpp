#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include "tensor.hpp"
namespace ops {
    void add(const Tensor& a, const Tensor& b, Tensor& output);
    void multiply(const Tensor& a, const Tensor& b, Tensor& output);
    void relu(const Tensor& input, Tensor& output);

    void matmul_naive(const Tensor& a, const Tensor& b, Tensor& output);
    void matmul_ikj(const Tensor& a, const Tensor& b, Tensor& output);
    void matmul_tiled(const Tensor& a, const Tensor& b, Tensor& output, const size_t bs);
    void matmul_neon(const Tensor& a, const Tensor& b, Tensor& output);
}

#endif
