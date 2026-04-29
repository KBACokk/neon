#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <arm_neon.h>
#include <iomanip>
#include <string>

// --- Вычислительные ядра ---

/** 
 * Скалярная версия: Сумма абсолютных значений положительных и отрицательных чисел.
 * Игнорирует нули.
 */
int64_t process_array_scalar(const int32_t* data, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        const int32_t val = data[i];
        if (val > 0)      sum += static_cast<int64_t>(val);
        else if (val < 0) sum += static_cast<int64_t>(-val);
    }
    return sum;
}

/** 
 * Векторная версия (ARM NEON): Использует SIMD для параллельной обработки 4-х чисел.
 */
int64_t process_array_neon(const int32_t* data, size_t n) {
    int32x4_t v_acc = vdupq_n_s32(0);
    size_t i = 0;

    // Основной цикл: обработка по 4 элемента
    for (; i + 3 < n; i += 4) {
        int32x4_t v_data = vld1q_s32(data + i);
        
        // Получаем абсолютное значение (vabsq_s32 эффективнее ручных манипуляций с масками)
        int32x4_t v_abs = vabsq_s32(v_data);
        
        // В оригинальной логике мы суммируем только если val != 0. 
        // Но abs(0) == 0, поэтому условие val != 0 можно опустить для скорости.
        v_acc = vaddq_s32(v_acc, v_abs);
    }

    // Горизонтальное сложение вектора в скаляр
    int64_t sum = vaddlvq_s32(v_acc);

    // Хвост массива (если n не кратно 4)
    for (; i < n; ++i) {
        const int32_t val = data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum -= static_cast<int64_t>(val);
    }

    return sum;
}

// --- Инструменты тестирования ---

struct BenchResult {
    size_t size;
    int64_t result;
    double t_scalar;
    double t_neon;
};

void print_header() {
    std::string line = "+" + std::string(12, '-') + "+" + std::string(14, '-') + "+" + std::string(14, '-') + "+" + std::string(14, '-') + "+" + std::string(12, '-') + "+";
    std::cout << line << "\n"
              << "| " << std::left << std::setw(10) << "Elements"
              << " | " << std::setw(12) << "Sum Result"
              << " | " << std::setw(12) << "Scalar (ms)"
              << " | " << std::setw(12) << "NEON (ms)"
              << " | " << std::setw(10) << "Boost" << " |\n"
              << line << "\n";
}

int main() {
    const std::vector<size_t> test_sizes = {1000, 10000, 100000, 1000000};
    
    // Настройка генератора
    std::mt19937 gen(42); // Фиксированный сид для воспроизводимости
    std::uniform_int_distribution<int32_t> dis(-10, 10);

    print_header();

    for (const auto n : test_sizes) {
        std::vector<int32_t> data(n);
        for (auto& x : data) x = dis(gen);

        // Бенчмарк Скаляра
        auto s_start = std::chrono::high_resolution_clock::now();
        volatile int64_t res_s = process_array_scalar(data.data(), n);
        auto s_end = std::chrono::high_resolution_clock::now();

        // Бенчмарк NEON
        auto n_start = std::chrono::high_resolution_clock::now();
        volatile int64_t res_n = process_array_neon(data.data(), n);
        auto n_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> ms_s = s_end - s_start;
        std::chrono::duration<double, std::milli> ms_n = n_end - n_start;

        // Вывод строки данных
        double speedup = ms_s.count() / ms_n.count();
        
        std::cout << "| " << std::left << std::setw(10) << n
                  << " | " << std::setw(12) << res_s
                  << " | " << std::setw(12) << std::fixed << std::setprecision(4) << ms_s.count()
                  << " | " << std::setw(12) << ms_n.count()
                  << " | " << std::setw(9) << std::setprecision(2) << speedup << "x" << " |\n";
    }

    std::cout << "+" + std::string(12, '-') + "+" + std::string(14, '-') + "+" + std::string(14, '-') + "+" + std::string(14, '-') + "+" + std::string(12, '-') + "+\n";

    return 0;
}
