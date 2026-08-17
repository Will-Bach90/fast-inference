#ifndef RUNTIME_HPP
#define RUNTIME_HPP

#include "tensor.hpp"

namespace runtime {
    class Runtime {
        public:
            void add(const Tensor& a, const Tensor& b, Tensor& output);
            void multiply(const Tensor& a, const Tensor& b, Tensor& output);
            void relu(const Tensor& input, Tensor& output);
            void matmul(const Tensor& a, const Tensor& b, Tensor& output);
    };
}


#endif