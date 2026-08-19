#include "operator.hpp"

namespace ops {
    void add(const Tensor& a, const Tensor& b, Tensor& output) {
        if(a.shape() != b.shape()) {
            throw std::invalid_argument("Input tensors must have the same shape");
        }

        if(a.shape() != output.shape()) {
            throw std::invalid_argument("Output tensor must have the correct shape");
        }

        // for(size_t i = 0; i < a.size(); ++i) {
        //     output.data()[i] = a.data()[i] + b.data()[i];
        // }

        const float* a_data = a.data();
        const float* b_data = b.data();
        float* output_data = output.data();

        const size_t size = a.size();

        for(size_t i = 0; i < size; ++i) {
            output_data[i] = a_data[i] + b_data[i];
        }
    }


    void relu(const Tensor& input, Tensor& output) {
        if(input.shape() != output.shape()) {
            throw std::invalid_argument("Output tensor must have the correct shape");
        }

        for(size_t i = 0; i< input.size(); ++i) {
            if(input.data()[i] < 0) {
                output.data()[i] = 0;
            } else {
                output.data()[i] = input.data()[i];
            }
        }
    }

    void multiply(const Tensor& a, const Tensor& b, Tensor& output) {
        if(a.shape() != b.shape()) {
            throw std::invalid_argument("Input tensors must have the same shape");
        }

        if(a.shape() != output.shape()) {
            throw std::invalid_argument("Output tensor must have the correct shape");
        }

        for(size_t i = 0; i < a.size(); ++i) {
            output.data()[i] = a.data()[i] * b.data()[i];
        }
    }

    void matmul(const Tensor& a, const Tensor& b, Tensor& output) {
        if(a.ndim() != 2 || b.ndim() != 2 || output.ndim() != 2) {
            throw std::invalid_argument("Tensors must be 2D");
        }

        // (M x K) * (K x N) = (M x N)
        const size_t M = a.shape()[0];
        const size_t K = a.shape()[1];

        const size_t K_b = b.shape()[0];
        const size_t N = b.shape()[1];

        if(K != K_b) {
            throw std::invalid_argument("Inner dimensions must match for matmul");
        }

        if(M != output.shape()[0] || N != output.shape()[1]) {
            throw std::invalid_argument("Output tensor must have the correct shape");
        }

        const float* a_data = a.data();
        const float* b_data = b.data();
        float* output_data = output.data();


        // for(size_t i = 0; i < M; ++i) {
        //     for(size_t k = 0; k < K; ++k) {
        //         float a_value = a_data[i*K+k];
        //         for(size_t j = 0; j < N; ++j) {
        //             output_data[i*N+j] += a_value * b_data[k*N+j];
        //         }
        //     }
        // }
        for(size_t i = 0; i < M; ++i) {
            for(size_t j = 0; j < N; j+=4) {
                float out0 = output_data[i*N+j + 0];
                float out1 = output_data[i*N+j + 1];
                float out2 = output_data[i*N+j + 2];
                float out3 = output_data[i*N+j + 3];

                for(size_t k = 0; k < K; ++k) {
                    float a_value = a_data[i*K + k];

                    out0 += a_value * b_data[k*N+j + 0];
                    out1 += a_value * b_data[k*N+j + 1];
                    out2 += a_value * b_data[k*N+j + 2];
                    out3 += a_value * b_data[k*N+j + 3];
                }

                output_data[i*N + j + 0] = out0;
                output_data[i*N + j + 1] = out1;
                output_data[i*N + j + 2] = out2;
                output_data[i*N + j + 3] = out3;
            }
        }
    }

}