#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>


class Tensor {
    float* data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
};

#endif // TENSOR_HPP