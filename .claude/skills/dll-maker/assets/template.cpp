#define DLL_EXPORT
#include "template.h"

#include <cstring>
#include <cmath>
#include <algorithm>

extern "C" {
    // === 기본 연산 구현 예시 ===
    /*
    API int add(int a, int b) {
        return a + b;
    }

    API double multiply(double a, double b) {
        return a * b;
    }
    */

    // === 배열 처리 구현 예시 ===
    /*
    API double sum_array(double* arr, size_t len) {
        double sum = 0.0;
        for (size_t i = 0; i < len; i++) {
            sum += arr[i];
        }
        return sum;
    }

    API void scale_array(double* arr, size_t len, double factor) {
        for (size_t i = 0; i < len; i++) {
            arr[i] *= factor;
        }
    }
    */

    // === 문자열 처리 구현 예시 ===
    /*
    API int process_string(const char* input, char* output, size_t output_size) {
        if (!input || !output || output_size == 0) {
            return -1;
        }

        size_t len = strlen(input);
        if (len >= output_size) {
            len = output_size - 1;
        }

        strncpy(output, input, len);
        output[len] = '\0';

        return static_cast<int>(len);
    }
    */

    // === 여기에 함수 구현 추가 ===
}
