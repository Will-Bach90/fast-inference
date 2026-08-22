#include "operator.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <functional>

int main() {

    // std::vector<size_t> sizes = {128, 256, 512, 1024};
    // const std::vector<size_t> block_sizes = {8, 16, 32, 64, 128, 256};

    // std::cout << std::left << std::setw(8) << 
    //     "M" << std::setw(8) << "N" << std::setw(8) 
    //     << "K" << std::setw(15) << "Time (s)" 
    //     << std::setw(15) << "GFLOPS" << "\n";

    std::cout << std::left << std::setw(26) << 
        "Implementation" << std::setw(20) << "1024 GFLOPS";
    
    std::cout << "\n";
    std::cout << std::string(40, '-') << "\n";

    constexpr size_t size = 1024;

    Tensor a({size, size});
    Tensor b({size, size});
    Tensor output({size, size});

    for (size_t j = 0; j < size*size; ++j) {
        a.data()[j] = 1.0f;
        b.data()[j] = 1.0f;
    }

    std::fill(output.data(), output.data() + output.size(), 0.0f);

    std::vector<std::pair<std::string, std::function<void(const Tensor&, const Tensor&, Tensor&)>>> implementations = {
        // {"Naive", ops::matmul_naive},
        {"IKJ", ops::matmul_ikj},
        {"NEON 4x4", ops::matmul_4x4},
        // {"Tiled (128)", [](const Tensor& a, const Tensor& b, Tensor& output) { ops::matmul_tiled(a, b, output, 128); }},
        // {"NEON", ops::matmul_neon},
        // {"NEON IKJ", ops::matmul_neon_ikj}
    };

    for(const auto& [name, func] : implementations) {
        std::fill(output.data(), output.data() + output.size(), 0.0f);

        auto start = std::chrono::steady_clock::now();

        for(int i = 0; i < 10; ++i) {
            std::fill(output.data(), output.data() + output.size(), 0.0f);
            func(a, b, output);
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end-start;
        double average_time = elapsed.count() / 10;
        double operations = 2.0 * size * size * size; // 2 * M * N * K for matrix multiplication
        double gflops = operations / (average_time *1e9); // GFLOPS

        std::cout << std::left << std::setw(26) << 
        name << std::setw(20) << std::fixed << std::setprecision(3) << gflops << "\n";

        float expected = static_cast<float>(size);

        for (size_t i = 0; i < size * size; ++i) {
            if (std::abs(output.data()[i] - expected) > 1e-5f) {
                std::cerr << "Incorrect result\n";
                return 1;
            }
        }

    }


    // for(const size_t bs : block_sizes) {
    //     std::cout << std::left << std::setw(8) << bs;
    //     std::cout << std::right << std::setw(3) << "|";
    //     for(size_t size : sizes) {
    //         if(bs > size) {
    //             continue; 
    //         }
    //         Tensor a({size, size});
    //         Tensor b({size, size});
    //         Tensor output({size, size});

    //         for (size_t j = 0; j < size*size; ++j) {
    //             a.data()[j] = 1.0f;
    //             b.data()[j] = 1.0f;
    //         }

    //         std::fill(output.data(), output.data() + output.size(), 0.0f);
    //         ops::matmul_tiled(a, b, output, bs);

    //         auto start = std::chrono::steady_clock::now();

    //         for(int i = 0; i < 10; ++i) {
    //             std::fill(output.data(), output.data() + output.size(), 0.0f);
    //             ops::matmul_tiled(a, b, output, bs);
    //         }

    //         auto end = std::chrono::steady_clock::now();

    //         std::chrono::duration<double> elapsed = end-start;

    //         double average_time = elapsed.count() / 10;

    //         double operations = 2.0 * size * size * size; // 2 * M * N * K for matrix multiplication
    //         double gflops = operations / (average_time * 1e9); // GFLOPS

    //         std::cout << std::left << std::setw(10) << gflops;
    //         std::cout << std::right << std::setw(3) << "|";

    //         float expected = static_cast<float>(size);

    //         for (size_t i = 0; i < size * size; ++i) {
    //             if (std::abs(output.data()[i] - expected) > 1e-5f) {
    //                 std::cerr << "Incorrect result\n";
    //                 return 1;
    //             }
    //         }
    //     }
    //     std::cout << "\n";

        // std::fill(output.data(), output.data() + output.size(), 0.0f);
        // ops::matmul(a, b, output, 64);

        // auto start = std::chrono::steady_clock::now();

        // for(int i = 0; i < 10; ++i) {
        //     std::fill(output.data(), output.data() + output.size(), 0.0f);
        //     ops::matmul(a, b, output, 64);
        // }

        // auto end = std::chrono::steady_clock::now();

        // std::chrono::duration<double> elapsed = end-start;

        // double average_time = elapsed.count() / 10;

        // double operations = 2.0 * size * size * size; // 2 * M * N * K for matrix multiplication
        // double gflops = operations / (average_time * 1e9); // GFLOPS

        // std::cout << std::left << std::setw(8) << size << std::setw(8) << size << std::setw(8) << size << std::setw(15) << average_time << std::setw(15) << gflops << "\n";

        // float expected = static_cast<float>(size);

        // for (size_t i = 0; i < size * size; ++i) {
        //     if (std::abs(output.data()[i] - expected) > 1e-5f) {
        //         std::cerr << "Incorrect result\n";
        //         return 1;
        //     }
        // }
    // }
    return 0;
}