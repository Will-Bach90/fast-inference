#include "operator.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>

int main() {

    size_t M = 512;
    size_t N = 512;    
    size_t K = 512;

    Tensor A({M, K});
    Tensor B({K, N});
    Tensor C({M, N});

    for (size_t i = 0; i < A.size(); ++i) {
        A.data()[i] = 1.0f;
        B.data()[i] = 1.0f;
    }

    std::cout << std::left << std::setw(15) << "N" << std::setw(15) << "time" << std::setw(15) << "gflops\n";
    std::cout << "---------------------------------------------\n";

    ops::matmul(A, B, C);

    for(int i = 0; i < 3; ++i) {
        ops::matmul(A, B, C);
    }

    auto start = std::chrono::steady_clock::now();

    for(int i = 0; i < 10; ++i) {
        ops::matmul(A, B, C);
    }

    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed = end-start;

    double average_time = elapsed.count() / 10;

    double operations = 2.0 * M * N * K; // 2 * M * N * K for matrix multiplication
    double gflops = (2.0*M*N*K) / (average_time *1000 * 1e9); // GFLOPS

    std::cout << std::left << std::setw(15) << N << std::setw(15) << average_time*1000 << std::setw(15) << gflops << "\n";

    return 0;
}