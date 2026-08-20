#include "operator.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {

    std::vector<size_t> sizes = {128, 256, 512, 1024};
    const std::vector<size_t> block_sizes = {8, 16, 32, 64, 128, 256};

    // std::cout << std::left << std::setw(8) << 
    //     "M" << std::setw(8) << "N" << std::setw(8) 
    //     << "K" << std::setw(15) << "Time (s)" 
    //     << std::setw(15) << "GFLOPS" << "\n";

    std::cout << std::left << std::setw(10) << 
        "Block Size" << "|"; 
    
    for(size_t size : sizes) {
        std::cout << std::left << std::setw(12) <<  std::to_string(size) << "|"; 
    }
    std::cout << "\n";
    std::cout << std::string(10, '-') << "|" 
              << std::string(12, '-') << "|" 
              << std::string(12, '-') << "|" 
              << std::string(12, '-') << "|" 
              << std::string(12, '-') << "|" 
              << "\n";

    for(const size_t bs : block_sizes) {
        std::cout << std::left << std::setw(8) << bs;
        std::cout << std::right << std::setw(3) << "|";
        for(size_t size : sizes) {
            if(bs > size) {
                continue; 
            }
            Tensor a({size, size});
            Tensor b({size, size});
            Tensor output({size, size});

            for (size_t j = 0; j < size*size; ++j) {
                a.data()[j] = 1.0f;
                b.data()[j] = 1.0f;
            }

            std::fill(output.data(), output.data() + output.size(), 0.0f);
            ops::matmul(a, b, output, bs);

            auto start = std::chrono::steady_clock::now();

            for(int i = 0; i < 10; ++i) {
                std::fill(output.data(), output.data() + output.size(), 0.0f);
                ops::matmul(a, b, output, bs);
            }

            auto end = std::chrono::steady_clock::now();

            std::chrono::duration<double> elapsed = end-start;

            double average_time = elapsed.count() / 10;

            double operations = 2.0 * size * size * size; // 2 * M * N * K for matrix multiplication
            double gflops = operations / (average_time * 1e9); // GFLOPS

            std::cout << std::left << std::setw(10) << gflops;
            std::cout << std::right << std::setw(3) << "|";

            float expected = static_cast<float>(size);

            for (size_t i = 0; i < size * size; ++i) {
                if (std::abs(output.data()[i] - expected) > 1e-5f) {
                    std::cerr << "Incorrect result\n";
                    return 1;
                }
            }
        }
        std::cout << "\n";

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
    }
    return 0;
}