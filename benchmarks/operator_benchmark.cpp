#include "operator.hpp"

#include <chrono>
#include <iostream>

int main() {
    constexpr size_t N = 10'000'000;

    Tensor a({N});
    Tensor b({N});
    Tensor output({N});

    for (size_t i = 0; i < N; ++i) {
        a.data()[i] = 1.0f;
        b.data()[i] = 2.0f;
    }

    ops::add(a, b, output);

    constexpr int iterations = 20;

    auto start = std::chrono::steady_clock::now();

    for(int i = 0; i < iterations; ++i) {
        ops::add(a, b, output);
    }

    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed = end-start;

    double average_time = elapsed.count() / iterations;

    std::cout << "Elements: " << N << "\n";
    std::cout << "Average time: " << average_time*1000 << " ms\n";

    return 0;
}