#pragma once

#ifdef DLL_EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif

#include <cstdint>
#include <cstddef>

extern "C" {
    // === 기본 연산 ===
    // API int add(int a, int b);
    // API double multiply(double a, double b);

    // === 배열 처리 ===
    // API double sum_array(double* arr, size_t len);
    // API void scale_array(double* arr, size_t len, double factor);

    // === 문자열 처리 ===
    // API int process_string(const char* input, char* output, size_t output_size);

    // === 여기에 함수 선언 추가 ===
}
