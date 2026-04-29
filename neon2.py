#include <iostream>
#include <vector>
#include <cstdint>
#include <arm_neon.h>
#include <chrono>

int64_t process_array_scalar(const int32_t* data, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) {
            sum += val;
        } else if (val < 0) {
            sum += (val == INT32_MIN) ? (int64_t)2147483648LL : -val;
        }
    }
    return sum;
}

int64_t process_array_neon(const int32_t* data, size_t n) {
    int64_t total_sum = 0;
    int32x4_t acc = vdupq_n_s32(0);
    
    size_t i = 0;
    int32x4_t zero_vec = vdupq_n_s32(0);

    for (; i + 3 < n; i += 4) {
        __builtin_prefetch(data + i + 8);

        int32x4_t vec = vld1q_s32(data + i);

        uint32x4_t mask_pos = vcgtq_s32(vec, zero_vec); // vec > 0
        int32x4_t mask_neg = vcltq_s32(vec, zero_vec); // vec < 0

        int32x4_t sign = vshrq_n_s32(vec, 31);
        int32x4_t abs_val = veorq_s32(vec, sign);
        abs_val = vsubq_s32(abs_val, sign);

       int32x4_t pos_part = vandq_s32(vec, mask_pos);
        int32x4_t neg_part = vandq_s32(abs_val, mask_neg);
        
        int32x4_t contrib = vorrq_s32(pos_part, neg_part);

        acc = vaddq_s32(acc, contrib);
    }

#if defined(__aarch64__)
    total_sum = vaddlvq_s32(acc);
#else

    int32_t temp[4];
    vst1q_s32(temp, acc);
    total_sum = (int64_t)temp[0] + temp[1] + temp[2] + temp[3];
#endif

    for (; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) total_sum += val;
        else if (val < 0) total_sum += (val == INT32_MIN) ? 2147483648LL : -val;
    }

    return total_sum;
}

int main() {
    const size_t N = 10000;
    alignas(16) int32_t data[N];

    for (size_t i = 0; i < N; ++i) {
        data[i] = (rand() % 201) - 100; 
    }

    auto start_s = std::chrono::high_resolution_clock::now();
    int64_t res_s = process_array_scalar(data, N);
    auto end_s = std::chrono::high_resolution_clock::now();

    auto start_v = std::chrono::high_resolution_clock::now();
    int64_t res_v = process_array_neon(data, N);
    auto end_v = std::chrono::high_resolution_clock::now();

    std::cout << "Scalar result: " << res_s << std::endl;
    std::cout << "NEON result:   " << res_v << std::endl;
    
    if (res_s == res_v) {
        std::cout << "Verification: SUCCESS" << std::endl;
    } else {
        std::cout << "Verification: FAILED" << std::endl;
    }

    return 0;
}
