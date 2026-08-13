#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>

class Tensor {
    public:
        Tensor(std::vector<size_t> shape);
        float* data();
        const float* data() const;

        const std::vector<size_t>& shape() const;
        const std::vector<size_t>& strides() const;

        size_t ndim() const;
        size_t size() const;

        float& at(const std::vector<size_t>& indices);
        const float& at(const std::vector<size_t>& indices) const;

    private:
        std::vector<float> data_;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
};

#endif // TENSOR_HPP