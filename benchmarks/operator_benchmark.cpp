#include "operator.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>

// / ====================================================================
// / Benchmarking code for the add operator
int main() {
    const std::vector<size_t> sizes = {
        1'000,
        10'000,
        100'000,
        1'000'000,
        10'000'000,
        100'000'000,
    };

    std::cout << std::left << std::setw(15) << "N" << std::setw(15) << "time" << std::setw(15) << "bandwidth\n";
    std::cout << "---------------------------------------------\n";

    constexpr int iterations = 20;

    for(int i = 0; i < sizes.size(); ++i) {
        size_t N = sizes[i];
        Tensor a({N});
        Tensor b({N});
        Tensor output({N});

        for (size_t j = 0; j < N; ++j) {
            a.data()[j] = 1.0f;
            b.data()[j] = 2.0f;
        }

        ops::add(a, b, output);

        auto start = std::chrono::steady_clock::now();

        for(int i = 0; i < iterations; ++i) {
            ops::add(a, b, output);
        }

        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed = end-start;

        double average_time = elapsed.count() / iterations;

        double bytes_processed = static_cast<double>(N)*3*sizeof(float);
        double bandwidth = bytes_processed/average_time/1e9; // GB/s

        std::cout << std::left << std::setw(15) << N << std::setw(15) << average_time*1000 << std::setw(15) << bandwidth << "\n";
    }

    return 0;
}