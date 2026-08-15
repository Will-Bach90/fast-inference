#include "tensor.hpp"

#include <stdexcept>
#include <utility>

Tensor::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
    if(shape_.empty()) {
        throw std::invalid_argument("Shape cannot be empty");
    }
    strides_.resize(shape_.size());
    size_t stride = 1;
    for (size_t i = shape_.size(); i-- > 0;) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
    data_.resize(stride);
}

size_t Tensor::offset(const std::vector<size_t>& indices) const {
    if(indices.size() != shape_.size()) {
        throw std::invalid_argument("Incorrect # of indices");
    }
    size_t offset = 0;

    for( size_t i = 0; i < indices.size(); i++) {
        if(indices[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of range");
        }
        offset+= indices[i]*strides_[i];
    }

    return offset;
}

float& Tensor::at(const std::vector<size_t>& indices) {
    return data_[offset(indices)];
}

float* Tensor::data() {
    return data_.data();
}

const float* Tensor::data() const {
    return data_.data();
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

const std::vector<size_t>& Tensor::strides() const {
    return strides_;
}

size_t Tensor::ndim() const {
    return shape_.size();
}

size_t Tensor::size() const {
    return data_.size();
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    return data_[offset(indices)];
}