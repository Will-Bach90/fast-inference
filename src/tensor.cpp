#include "../include/tensor.hpp"

Tensor::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
    // Calculate strides
    strides_.resize(shape_.size());
    size_t stride = 1;
    for (size_t i = shape_.size(); i-- > 0;) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
    data_.resize(stride);
}