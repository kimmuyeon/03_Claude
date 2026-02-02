---
name: dll-maker
description: 사용자가 "dll-maker", "/dll-maker", "dll-maker 스킬" 등으로 명시적으로 이 스킬을 언급할 때만 사용합니다. Python 코드를 C++ 기반 Windows DLL로 변환하는 스킬입니다.
---

# DLL Maker

Python 로직을 C++ 기반 Windows DLL로 변환하는 스킬입니다.

## 참조 문서

| 문서 | 설명 |
|------|------|
| `references/advanced_patterns.md` | 콜백, 에러 처리, 메모리 관리, SIMD, GIL 해제 |
| `references/ai_model_patterns.md` | 모델 라이프사이클, 가중치 로딩, 배치 처리, GPU/CUDA, ONNX Runtime, TensorRT |
| `references/testing_validation.md` | Python-C++ 결과 검증, pytest, Google Test |

## Workflow

```
사용자 요청 분석
    ↓
┌─────────────────────────────────────┐
│ Python 코드 분석                      │
│ - 함수 시그니처 파악                   │
│ - 데이터 타입 매핑                     │
│ - 의존성 확인                         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ C++ 코드 생성                         │
│ - 헤더 파일 (.h)                      │
│ - 구현 파일 (.cpp)                    │
│ - DLL export 선언                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 빌드 설정 생성                        │
│ - CMakeLists.txt                     │
│ - 또는 Visual Studio 프로젝트          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Python 래퍼 코드 생성                  │
│ - ctypes 기반 호출 코드                │
│ - 사용 예제                           │
└─────────────────────────────────────┘
```

## 타입 매핑 규칙

| Python | C++ | ctypes |
|--------|-----|--------|
| `int` | `int` / `int64_t` | `c_int` / `c_int64` |
| `float` | `double` | `c_double` |
| `bool` | `bool` | `c_bool` |
| `str` | `const char*` | `c_char_p` |
| `bytes` | `const uint8_t*` + `size_t` | `c_char_p` + `c_size_t` |
| `List[int]` | `int*` + `size_t` | `POINTER(c_int)` + `c_size_t` |
| `List[float]` | `double*` + `size_t` | `POINTER(c_double)` + `c_size_t` |
| `numpy.ndarray` | `double*` + shape 정보 | `np.ctypeslib.ndpointer` |

## C++ DLL 템플릿

### 헤더 파일 (.h)
```cpp
#pragma once

#ifdef DLL_EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif

extern "C" {
    // 함수 선언
    API int example_function(int a, int b);
    API double process_array(double* arr, size_t len);
}
```

### 구현 파일 (.cpp)
```cpp
#define DLL_EXPORT
#include "example.h"

extern "C" {
    API int example_function(int a, int b) {
        return a + b;
    }

    API double process_array(double* arr, size_t len) {
        double sum = 0.0;
        for (size_t i = 0; i < len; i++) {
            sum += arr[i];
        }
        return sum;
    }
}
```

### CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.10)
project(MyDLL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(mydll SHARED
    src/mydll.cpp
)

target_include_directories(mydll PUBLIC include)
```

## Python ctypes 래퍼 템플릿

```python
import ctypes
import numpy as np
from pathlib import Path

class MyDLL:
    def __init__(self, dll_path: str = None):
        if dll_path is None:
            dll_path = Path(__file__).parent / "mydll.dll"
        self._dll = ctypes.CDLL(str(dll_path))
        self._setup_functions()

    def _setup_functions(self):
        # example_function
        self._dll.example_function.argtypes = [ctypes.c_int, ctypes.c_int]
        self._dll.example_function.restype = ctypes.c_int

        # process_array
        self._dll.process_array.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_size_t
        ]
        self._dll.process_array.restype = ctypes.c_double

    def example_function(self, a: int, b: int) -> int:
        return self._dll.example_function(a, b)

    def process_array(self, arr: np.ndarray) -> float:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        return self._dll.process_array(arr, len(arr))
```

## 빌드 방법

### CMake (권장)
```bash
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Visual Studio Developer Command Prompt
```bash
cl /LD /EHsc /O2 /Fe:mydll.dll src/mydll.cpp
```

## 변환 시 주의사항

1. **메모리 관리**: Python GC와 달리 C++은 수동 메모리 관리 필요
2. **예외 처리**: C++ 예외가 DLL 경계를 넘지 않도록 `extern "C"` 내부에서 처리
3. **스레드 안전성**: 전역 변수 사용 시 mutex 고려
4. **데이터 정렬**: 구조체 전달 시 `#pragma pack` 확인
5. **호출 규약**: Windows 기본은 `__cdecl`, 명시적으로 지정 권장

## 복잡한 데이터 구조 처리

### 구조체 전달
```cpp
// C++
#pragma pack(push, 1)
struct Point {
    double x;
    double y;
};
#pragma pack(pop)

extern "C" API double distance(Point* p1, Point* p2);
```

```python
# Python
class Point(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]
```

### 문자열 반환
```cpp
// C++ - 버퍼 방식 (권장)
extern "C" API int get_string(char* buffer, size_t buffer_size);
```

```python
# Python
buffer = ctypes.create_string_buffer(256)
dll.get_string(buffer, 256)
result = buffer.value.decode('utf-8')
```

## AI 모델 변환 시 추가 고려사항

AI 모델을 DLL로 변환할 때는 추가적인 고려가 필요합니다. 자세한 내용은 `references/ai_model_patterns.md` 참조.

### 모델 라이프사이클
```cpp
// 필수 패턴: init/cleanup 쌍
extern "C" {
    API ModelHandle model_init(const char* config_path);
    API void model_cleanup(ModelHandle handle);
    API int model_predict(ModelHandle handle, float* input, float* output);
}
```

### 가중치 로딩 방식 선택
| 방식 | 장점 | 단점 |
|------|------|------|
| 바이너리 파일 | 빠른 로딩, 작은 크기 | 커스텀 포맷 필요 |
| ONNX Runtime | 표준 포맷, GPU 지원 | 추가 의존성 |
| 하드코딩 | 단일 파일 배포 | 모델 업데이트 어려움 |

### 배치 처리 권장
```cpp
// 단일 처리보다 배치 처리가 효율적
API int batch_predict(
    float* images,      // N개 이미지 연속 배치
    size_t batch_size,
    float* outputs      // N개 결과
);
```

### Python 멀티스레드 환경
- ctypes.CDLL 사용 시 GIL 자동 해제
- 콜백 함수 사용 시 GIL 재획득 주의
- 자세한 내용: `references/advanced_patterns.md`의 GIL 해제 패턴 참조

### 검증 필수
Python 원본과 C++ DLL의 결과가 일치하는지 반드시 검증:
```python
# 허용 오차 내 일치 확인
np.testing.assert_allclose(python_result, cpp_result, rtol=1e-5, atol=1e-8)
```
자세한 테스트 방법: `references/testing_validation.md` 참조
