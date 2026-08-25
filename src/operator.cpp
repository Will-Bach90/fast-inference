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

    static inline float32x4_t matmul_kernel(const float* a, const float* b, float32x4_t c, size_t K, size_t N) {
        for(size_t k = 0; k < K; ++k) {
            float a_value = a[k];
            float32x4_t b_vector = vld1q_f32(b + k*N);

            c = vmlaq_n_f32(c, b_vector, a_value);
        }
        return c;
    }

    void matmul_naive(const Tensor& a, const Tensor& b, Tensor& output) {
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

        for(size_t i = 0; i < M; ++i) {
            for(size_t j = 0; j < N; ++j) {
                for(size_t k = 0; k < K; ++k) {
                    output_data[i*N+j] += a_data[i*K+k] * b_data[k*N+j];
                }
            }
        }
    }
    void matmul_ikj(const Tensor& a, const Tensor& b, Tensor& output) {
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
        
        for(size_t i = 0; i < M; ++i) {
            for(size_t k = 0; k < K; ++k) {
                float a_value = a_data[i*K+k];
                for(size_t j = 0; j < N; ++j) {
                    output_data[i*N+j] += a_value * b_data[k*N+j];
                }
            }
        }
    }

    void matmul_tiled(const Tensor& a, const Tensor& b, Tensor& output, const size_t bs) {
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

        for(size_t ii = 0; ii < M; ii += bs) {
            for(size_t kk = 0; kk < K; kk += bs) {
                for(size_t jj = 0; jj < N; jj += bs) {

                    for(size_t i = ii; i < ii + bs; ++i) {
                        for(size_t k = kk; k < kk + bs; ++k) {
                            float a_value = a_data[i*K+k];
                            for(size_t j = jj; j < jj + bs; ++j) {
                                output_data[i*N+j] += a_value * b_data[k*N+j];
                            }
                        }
                    }

                }
            }
        }
    }


    void matmul_neon(const Tensor& a, const Tensor& b, Tensor& output) {
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

        for(size_t i = 0; i < M; ++i) {
            for(size_t j = 0; j < N; j += 4) {
                float32x4_t c = vld1q_f32(output_data + i*N + j);

                c = matmul_kernel(a_data+i*K, b_data+j, c, K, N);

                vst1q_f32(output_data + i*N + j, c);
            }
        }
    }

    void matmul_neon_ikj(const Tensor& a, const Tensor& b, Tensor& output) {
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

        for(size_t i = 0; i < M; ++i) {
            for(size_t k = 0; k < K; ++k) {
                float a_value = a_data[i*K + k];


                float32x4_t a_vector = vdupq_n_f32(a_value);

                for(size_t j = 0; j < N; j +=4) {
                    float32x4_t b_vector = vld1q_f32(b_data + k*N + j);
                    float32x4_t c_vector = vld1q_f32(output_data + i*N + j);

                    c_vector = vmlaq_f32(c_vector, a_vector, b_vector);

                    vst1q_f32(output_data + i*N + j, c_vector);
                }
            }
        }
    }


    static inline void matmul_kernel_4x4(const float* a, const float* b, float* c, size_t K, size_t N) {
        float32x4_t c0 = vld1q_f32(c);
        float32x4_t c1 = vld1q_f32(c + N);
        float32x4_t c2 = vld1q_f32(c + 2*N);
        float32x4_t c3 = vld1q_f32(c + 3*N);

        for(size_t k = 0; k < K; ++k) {
            float32x4_t b_vec = vld1q_f32(b + k*N);
            
            c0 = vmlaq_n_f32(c0, b_vec, a[k]);
            c1 = vmlaq_n_f32(c1, b_vec, a[N+k]);
            c2 = vmlaq_n_f32(c2, b_vec, a[2*N+k]);
            c3 = vmlaq_n_f32(c3, b_vec, a[3*N+k]);

        }

        vst1q_f32(c, c0);
        vst1q_f32(c + N, c1);
        vst1q_f32(c + 2*N, c2);
        vst1q_f32(c + 3*N, c3);
    }

    void matmul_4x4(const Tensor& a, const Tensor& b, Tensor& output) {
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

        for(size_t i = 0; i < M; i+=4) {
            for(size_t j = 0; j < N; j+=4) {
                matmul_kernel_4x4(a_data + i*K, b_data+j, output_data + i*N+j, K, N);
            }
        }
    }

    static inline void matmul_kernel_8x4(const float* a, const float* b, float* c, size_t K, size_t N) {
        float32x4_t c0 = vld1q_f32(c);
        float32x4_t c1 = vld1q_f32(c + N);
        float32x4_t c2 = vld1q_f32(c + 2*N);
        float32x4_t c3 = vld1q_f32(c + 3*N);
        float32x4_t c4 = vld1q_f32(c + 4*N);
        float32x4_t c5 = vld1q_f32(c + 5*N);
        float32x4_t c6 = vld1q_f32(c + 6*N);
        float32x4_t c7 = vld1q_f32(c + 7*N);

        for(size_t k = 0; k < K; ++k) {
            float32x4_t b_vec = vld1q_f32(b + k*N);
            
            c0 = vmlaq_n_f32(c0, b_vec, a[k]);
            c1 = vmlaq_n_f32(c1, b_vec, a[N+k]);
            c2 = vmlaq_n_f32(c2, b_vec, a[2*N+k]);
            c3 = vmlaq_n_f32(c3, b_vec, a[3*N+k]);
            c4 = vmlaq_n_f32(c4, b_vec, a[4*N+k]);
            c5 = vmlaq_n_f32(c5, b_vec, a[5*N+k]);
            c6 = vmlaq_n_f32(c6, b_vec, a[6*N+k]);
            c7 = vmlaq_n_f32(c7, b_vec, a[7*N+k]);

        }

        vst1q_f32(c, c0);
        vst1q_f32(c + N, c1);
        vst1q_f32(c + 2*N, c2);
        vst1q_f32(c + 3*N, c3);
        vst1q_f32(c + 4*N, c4);
        vst1q_f32(c + 5*N, c5);
        vst1q_f32(c + 6*N, c6);
        vst1q_f32(c + 7*N, c7);
    }


    void matmul_8x4(const Tensor& a, const Tensor& b, Tensor& output) {
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

        for(size_t i = 0; i < M; i+=8) {
            for(size_t j = 0; j < N; j+=4) {
                matmul_kernel_8x4(a_data + i*K, b_data+j, output_data + i*N+j, K, N);
            }
        }
    }




    // ===========================≠≠≠≠≠≠≠=============++++++++==============================
    static inline void matmul_kernel_12x4(const float* a, const float* b, float* c, size_t K, size_t N) {
        float32x4_t c0 = vld1q_f32(c);
        float32x4_t c1 = vld1q_f32(c + N);
        float32x4_t c2 = vld1q_f32(c + 2*N);
        float32x4_t c3 = vld1q_f32(c + 3*N);
        float32x4_t c4 = vld1q_f32(c + 4*N);
        float32x4_t c5 = vld1q_f32(c + 5*N);
        float32x4_t c6 = vld1q_f32(c + 6*N);
        float32x4_t c7 = vld1q_f32(c + 7*N);
        float32x4_t c8 = vld1q_f32(c + 8*N);
        float32x4_t c9 = vld1q_f32(c + 9*N);
        float32x4_t c10 = vld1q_f32(c + 10*N);
        float32x4_t c11 = vld1q_f32(c + 11*N);

        for(size_t k = 0; k < K; ++k) {
            float32x4_t b_vec = vld1q_f32(b + k*N);
            
            c0 = vmlaq_n_f32(c0, b_vec, a[k]);
            c1 = vmlaq_n_f32(c1, b_vec, a[N+k]);
            c2 = vmlaq_n_f32(c2, b_vec, a[2*N+k]);
            c3 = vmlaq_n_f32(c3, b_vec, a[3*N+k]);
            c4 = vmlaq_n_f32(c4, b_vec, a[4*N+k]);
            c5 = vmlaq_n_f32(c5, b_vec, a[5*N+k]);
            c6 = vmlaq_n_f32(c6, b_vec, a[6*N+k]);
            c7 = vmlaq_n_f32(c7, b_vec, a[7*N+k]);
            c8 = vmlaq_n_f32(c8, b_vec, a[8*N+k]);
            c9 = vmlaq_n_f32(c9, b_vec, a[9*N+k]);
            c10 = vmlaq_n_f32(c10, b_vec, a[10*N+k]);
            c11 = vmlaq_n_f32(c11, b_vec, a[11*N+k]);

        }

        vst1q_f32(c, c0);
        vst1q_f32(c + N, c1);
        vst1q_f32(c + 2*N, c2);
        vst1q_f32(c + 3*N, c3);
        vst1q_f32(c + 4*N, c4);
        vst1q_f32(c + 5*N, c5);
        vst1q_f32(c + 6*N, c6);
        vst1q_f32(c + 7*N, c7);
        vst1q_f32(c + 8*N, c8);
        vst1q_f32(c + 9*N, c9);
        vst1q_f32(c + 10*N, c10);
        vst1q_f32(c + 11*N, c11);
    }


    void matmul_12x4(const Tensor& a, const Tensor& b, Tensor& output) {
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

        for(size_t i = 0; i < M; i+=12) {
            for(size_t j = 0; j < N; j+=4) {
                matmul_kernel_12x4(a_data + i*K, b_data+j, output_data + i*N+j, K, N);
            }
        }
    }

    // ===========================≠≠≠≠≠≠≠=============++++++++==============================
    static inline void matmul_kernel_14x4(const float* a, const float* b, float* c, size_t K, size_t N) {
        float32x4_t c0 = vld1q_f32(c);
        float32x4_t c1 = vld1q_f32(c + N);
        float32x4_t c2 = vld1q_f32(c + 2*N);
        float32x4_t c3 = vld1q_f32(c + 3*N);
        float32x4_t c4 = vld1q_f32(c + 4*N);
        float32x4_t c5 = vld1q_f32(c + 5*N);
        float32x4_t c6 = vld1q_f32(c + 6*N);
        float32x4_t c7 = vld1q_f32(c + 7*N);
        float32x4_t c8 = vld1q_f32(c + 8*N);
        float32x4_t c9 = vld1q_f32(c + 9*N);
        float32x4_t c10 = vld1q_f32(c + 10*N);
        float32x4_t c11 = vld1q_f32(c + 11*N);

        float32x4_t c12 = vld1q_f32(c + 12*N);
        float32x4_t c13 = vld1q_f32(c + 13*N);

        for(size_t k = 0; k < K; ++k) {
            float32x4_t b_vec = vld1q_f32(b + k*N);
            
            c0 = vmlaq_n_f32(c0, b_vec, a[k]);
            c1 = vmlaq_n_f32(c1, b_vec, a[N+k]);
            c2 = vmlaq_n_f32(c2, b_vec, a[2*N+k]);
            c3 = vmlaq_n_f32(c3, b_vec, a[3*N+k]);
            c4 = vmlaq_n_f32(c4, b_vec, a[4*N+k]);
            c5 = vmlaq_n_f32(c5, b_vec, a[5*N+k]);
            c6 = vmlaq_n_f32(c6, b_vec, a[6*N+k]);
            c7 = vmlaq_n_f32(c7, b_vec, a[7*N+k]);
            c8 = vmlaq_n_f32(c8, b_vec, a[8*N+k]);
            c9 = vmlaq_n_f32(c9, b_vec, a[9*N+k]);
            c10 = vmlaq_n_f32(c10, b_vec, a[10*N+k]);
            c11 = vmlaq_n_f32(c11, b_vec, a[11*N+k]);

            c12 = vmlaq_n_f32(c12, b_vec, a[12*N+k]);
            c13 = vmlaq_n_f32(c13, b_vec, a[13*N+k]);

        }

        vst1q_f32(c, c0);
        vst1q_f32(c + N, c1);
        vst1q_f32(c + 2*N, c2);
        vst1q_f32(c + 3*N, c3);
        vst1q_f32(c + 4*N, c4);
        vst1q_f32(c + 5*N, c5);
        vst1q_f32(c + 6*N, c6);
        vst1q_f32(c + 7*N, c7);
        vst1q_f32(c + 8*N, c8);
        vst1q_f32(c + 9*N, c9);
        vst1q_f32(c + 10*N, c10);
        vst1q_f32(c + 11*N, c11);

        vst1q_f32(c + 12*N, c12);
        vst1q_f32(c + 13*N, c13);
    }


    void matmul_14x4(const Tensor& a, const Tensor& b, Tensor& output) {
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

        for(size_t i = 0; i < M; i+=14) {
            for(size_t j = 0; j < N; j+=4) {
                matmul_kernel_14x4(a_data + i*K, b_data+j, output_data + i*N+j, K, N);
            }
        }
    }

    // ===========================≠≠≠≠≠≠≠=============++++++++==============================
    static inline void matmul_kernel_16x4(const float* a, const float* b, float* c, size_t K, size_t N) {
        float32x4_t c0 = vld1q_f32(c);
        float32x4_t c1 = vld1q_f32(c + N);
        float32x4_t c2 = vld1q_f32(c + 2*N);
        float32x4_t c3 = vld1q_f32(c + 3*N);
        float32x4_t c4 = vld1q_f32(c + 4*N);
        float32x4_t c5 = vld1q_f32(c + 5*N);
        float32x4_t c6 = vld1q_f32(c + 6*N);
        float32x4_t c7 = vld1q_f32(c + 7*N);
        float32x4_t c8 = vld1q_f32(c + 8*N);
        float32x4_t c9 = vld1q_f32(c + 9*N);
        float32x4_t c10 = vld1q_f32(c + 10*N);
        float32x4_t c11 = vld1q_f32(c + 11*N);

        float32x4_t c12 = vld1q_f32(c + 12*N);
        float32x4_t c13 = vld1q_f32(c + 13*N);
        float32x4_t c14 = vld1q_f32(c + 14*N);
        float32x4_t c15 = vld1q_f32(c + 15*N);

        for(size_t k = 0; k < K; ++k) {
            float32x4_t b_vec = vld1q_f32(b + k*N);
            
            c0 = vmlaq_n_f32(c0, b_vec, a[k]);
            c1 = vmlaq_n_f32(c1, b_vec, a[N+k]);
            c2 = vmlaq_n_f32(c2, b_vec, a[2*N+k]);
            c3 = vmlaq_n_f32(c3, b_vec, a[3*N+k]);
            c4 = vmlaq_n_f32(c4, b_vec, a[4*N+k]);
            c5 = vmlaq_n_f32(c5, b_vec, a[5*N+k]);
            c6 = vmlaq_n_f32(c6, b_vec, a[6*N+k]);
            c7 = vmlaq_n_f32(c7, b_vec, a[7*N+k]);
            c8 = vmlaq_n_f32(c8, b_vec, a[8*N+k]);
            c9 = vmlaq_n_f32(c9, b_vec, a[9*N+k]);
            c10 = vmlaq_n_f32(c10, b_vec, a[10*N+k]);
            c11 = vmlaq_n_f32(c11, b_vec, a[11*N+k]);

            c12 = vmlaq_n_f32(c12, b_vec, a[12*N+k]);
            c13 = vmlaq_n_f32(c13, b_vec, a[13*N+k]);
            c14 = vmlaq_n_f32(c14, b_vec, a[14*N+k]);
            c15 = vmlaq_n_f32(c15, b_vec, a[15*N+k]);

        }

        vst1q_f32(c, c0);
        vst1q_f32(c + N, c1);
        vst1q_f32(c + 2*N, c2);
        vst1q_f32(c + 3*N, c3);
        vst1q_f32(c + 4*N, c4);
        vst1q_f32(c + 5*N, c5);
        vst1q_f32(c + 6*N, c6);
        vst1q_f32(c + 7*N, c7);
        vst1q_f32(c + 8*N, c8);
        vst1q_f32(c + 9*N, c9);
        vst1q_f32(c + 10*N, c10);
        vst1q_f32(c + 11*N, c11);

        vst1q_f32(c + 12*N, c12);
        vst1q_f32(c + 13*N, c13);
        vst1q_f32(c + 14*N, c14);
        vst1q_f32(c + 15*N, c15);
    }


    void matmul_16x4(const Tensor& a, const Tensor& b, Tensor& output) {
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

        for(size_t i = 0; i < M; i+=16) {
            for(size_t j = 0; j < N; j+=4) {
                matmul_kernel_16x4(a_data + i*K, b_data+j, output_data + i*N+j, K, N);
            }
        }
    }

}