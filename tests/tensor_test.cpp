// #include "../include/operator.hpp"

// #include <iostream>

// int main() {
//     Tensor a({2, 3});
//     Tensor b({2, 3});
//     Tensor c({2, 3});

//     for (size_t i = 0; i < a.size(); ++i) {
//         a.data()[i] = static_cast<float>(i);
//         b.data()[i] = static_cast<float>(i * 10);
//     }

//     ops::add(a, b, c);

//     for (size_t i = 0; i < c.size(); ++i) {
//         std::cout << c.data()[i] << ' ';
//     }

//     std::cout << '\n';

//     return 0;
// }

#include "runtime.hpp"

#include <iostream>

int main() {
    runtime::Runtime rt;

    Tensor a({6});
    Tensor b({6});
    Tensor c({6});

    for (size_t i = 0; i < a.size(); ++i) {
        a.data()[i] = static_cast<float>(i);
        b.data()[i] = static_cast<float>(i * 10);
    }

    rt.add(a, b, c);

    for (size_t i = 0; i < c.size(); ++i) {
        std::cout << c.data()[i] << ' ';
    }

    std::cout << '\n';

    return 0;
}