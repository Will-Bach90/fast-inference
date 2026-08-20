#include "runtime.hpp"
#include "operator.hpp"

namespace runtime {
    void Runtime::add(const Tensor& a, const Tensor& b, Tensor& output) {
        ops::add(a, b, output);
    }

    void Runtime::multiply(const Tensor& a, const Tensor& b, Tensor& output) {
        ops::multiply(a, b, output);
    }

    void Runtime::relu(const Tensor& input, Tensor& output) {
        ops::relu(input, output);
    }

    // void Runtime::matmul(const Tensor& a, const Tensor& b, Tensor& output) {
    //     ops::matmul(a, b, output);
    // }

}