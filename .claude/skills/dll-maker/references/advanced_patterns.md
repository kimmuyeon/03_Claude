# 고급 패턴

## 콜백 함수 처리

### C++
```cpp
typedef void (*ProgressCallback)(int current, int total);

extern "C" API void process_with_callback(double* data, size_t len, ProgressCallback callback);
```

### Python
```python
CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int)

def progress_handler(current, total):
    print(f"Progress: {current}/{total}")

callback = CALLBACK_TYPE(progress_handler)
dll.process_with_callback(data, len(data), callback)
```

## 에러 핸들링 패턴

### C++
```cpp
enum ErrorCode {
    SUCCESS = 0,
    ERR_NULL_POINTER = -1,
    ERR_INVALID_SIZE = -2,
    ERR_OUT_OF_MEMORY = -3
};

extern "C" API int safe_operation(double* arr, size_t len) {
    if (!arr) return ERR_NULL_POINTER;
    if (len == 0) return ERR_INVALID_SIZE;

    try {
        // 연산 수행
        return SUCCESS;
    } catch (...) {
        return ERR_OUT_OF_MEMORY;
    }
}
```

### Python
```python
class DLLError(Exception):
    CODES = {
        -1: "Null pointer error",
        -2: "Invalid size error",
        -3: "Out of memory error"
    }

    def __init__(self, code):
        self.code = code
        super().__init__(self.CODES.get(code, f"Unknown error: {code}"))

def safe_operation(self, arr):
    result = self._dll.safe_operation(arr, len(arr))
    if result != 0:
        raise DLLError(result)
```

## 메모리 관리 패턴

### DLL에서 메모리 할당/해제
```cpp
extern "C" {
    API double* create_array(size_t len) {
        return new double[len];
    }

    API void free_array(double* arr) {
        delete[] arr;
    }
}
```

### Python에서 관리
```python
class ManagedArray:
    def __init__(self, dll, size):
        self._dll = dll
        self._ptr = dll.create_array(size)
        self._size = size

    def __del__(self):
        if self._ptr:
            self._dll.free_array(self._ptr)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._ptr:
            self._dll.free_array(self._ptr)
            self._ptr = None
```

## NumPy 배열 직접 전달

### 1D 배열
```cpp
extern "C" API void process_1d(double* data, size_t len);
```

```python
arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
dll.process_1d(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(arr))

# 또는 ndpointer 사용
dll.process_1d.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_size_t
]
dll.process_1d(arr, len(arr))
```

### 2D 배열
```cpp
extern "C" API void process_2d(double* data, size_t rows, size_t cols);
```

```python
arr = np.array([[1, 2], [3, 4]], dtype=np.float64, order='C')
dll.process_2d(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               arr.shape[0], arr.shape[1])
```

## 멀티스레딩

### Thread-safe 싱글톤
```cpp
#include <mutex>

class Processor {
    static Processor* instance;
    static std::mutex mtx;

public:
    static Processor* getInstance() {
        std::lock_guard<std::mutex> lock(mtx);
        if (!instance) {
            instance = new Processor();
        }
        return instance;
    }
};

extern "C" API void thread_safe_process(double* data, size_t len) {
    std::lock_guard<std::mutex> lock(global_mutex);
    // 연산 수행
}
```

## SIMD 최적화

```cpp
#include <immintrin.h>  // AVX

extern "C" API void simd_add(float* a, float* b, float* result, size_t len) {
    size_t i = 0;

    // AVX로 8개씩 처리
    for (; i + 8 <= len; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }

    // 나머지 처리
    for (; i < len; i++) {
        result[i] = a[i] + b[i];
    }
}
```

## Python GIL 해제 패턴

Python에서 DLL을 호출할 때 GIL(Global Interpreter Lock)을 해제하면 멀티스레드 환경에서 성능이 향상됩니다.

### 문제 상황

Python은 GIL로 인해 한 번에 하나의 스레드만 Python 바이트코드를 실행할 수 있습니다. DLL 호출 중에도 GIL이 유지되면 다른 Python 스레드가 블로킹됩니다.

```python
# GIL 미해제 시 문제
import threading
import time

def worker(dll, data):
    # 이 호출 중 다른 스레드가 블로킹됨
    dll.long_running_operation(data)

# 4개 스레드가 순차 실행됨 (병렬 아님)
threads = [threading.Thread(target=worker, args=(dll, data)) for _ in range(4)]
```

### 해결 방법 1: ctypes 직접 호출 시 GIL 해제

```python
import ctypes
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class GILFreeDLL:
    """GIL 해제를 지원하는 DLL 래퍼"""

    def __init__(self, dll_path: str):
        # CDLL 대신 PyDLL 사용하지 않음 (PyDLL은 GIL 유지)
        # CDLL은 기본적으로 GIL 해제
        self._dll = ctypes.CDLL(dll_path)
        self._setup_functions()

    def _setup_functions(self):
        # argtypes와 restype을 설정하면 ctypes가 자동으로 GIL 해제
        self._dll.process_data.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t
        ]
        self._dll.process_data.restype = ctypes.c_int

    def process_data(self, data: np.ndarray) -> int:
        """GIL이 자동으로 해제되는 호출"""
        data = np.ascontiguousarray(data, dtype=np.float32)
        return self._dll.process_data(data, len(data))


# 사용 예시: 진정한 병렬 처리
dll = GILFreeDLL("mymodel.dll")

def worker(data):
    return dll.process_data(data)

# ThreadPoolExecutor로 병렬 처리
with ThreadPoolExecutor(max_workers=4) as executor:
    data_chunks = [np.random.rand(10000).astype(np.float32) for _ in range(4)]
    results = list(executor.map(worker, data_chunks))
```

### 해결 방법 2: Cython을 통한 명시적 GIL 해제

```cython
# gil_free_wrapper.pyx
cimport cython
from libc.stddef cimport size_t

# DLL 함수 선언
cdef extern from "mymodel.h":
    int process_data(float* data, size_t length) nogil
    int batch_process(float* data, size_t batch_size, size_t item_size) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
def process_with_gil_release(float[::1] data):
    """GIL을 명시적으로 해제하고 DLL 호출"""
    cdef int result
    cdef size_t length = data.shape[0]

    # GIL 해제
    with nogil:
        result = process_data(&data[0], length)

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_process_parallel(float[:, ::1] batch_data):
    """배치 데이터를 GIL 해제 상태로 처리"""
    cdef int result
    cdef size_t batch_size = batch_data.shape[0]
    cdef size_t item_size = batch_data.shape[1]

    with nogil:
        result = batch_process(&batch_data[0, 0], batch_size, item_size)

    return result
```

```python
# setup.py for Cython
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("gil_free_wrapper.pyx"),
    include_dirs=[np.get_include()],
)
```

### 해결 방법 3: concurrent.futures와 ProcessPoolExecutor

GIL 문제를 완전히 우회하려면 멀티프로세싱 사용:

```python
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# 주의: DLL은 각 프로세스에서 별도로 로드됨
def process_in_worker(dll_path: str, data: np.ndarray) -> np.ndarray:
    """워커 프로세스에서 실행되는 함수"""
    import ctypes
    dll = ctypes.CDLL(dll_path)

    dll.process_data.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
        ctypes.c_size_t
    ]
    dll.process_data.restype = ctypes.c_int

    data = np.ascontiguousarray(data, dtype=np.float32)
    dll.process_data(data, len(data))
    return data

def parallel_process(dll_path: str, data_list: list) -> list:
    """멀티프로세스로 병렬 처리"""
    worker = partial(process_in_worker, dll_path)

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker, data_list))

    return results
```

### 해결 방법 4: NumPy 벡터화와 DLL 조합

단일 호출로 배치 처리하여 GIL 오버헤드 최소화:

```cpp
// C++ - 배치 처리 함수
extern "C" API int process_batch(
    float* data,           // 모든 데이터 연속 배치
    size_t* offsets,       // 각 아이템 오프셋
    size_t* lengths,       // 각 아이템 길이
    size_t num_items,      // 아이템 수
    float* results         // 결과 배열
) {
    #pragma omp parallel for  // OpenMP 병렬화
    for (size_t i = 0; i < num_items; i++) {
        float* item_data = data + offsets[i];
        size_t item_len = lengths[i];

        // 각 아이템 처리
        results[i] = compute_result(item_data, item_len);
    }
    return 0;
}
```

```python
# Python - 단일 호출로 배치 처리
class BatchProcessor:
    def __init__(self, dll_path: str):
        self._dll = ctypes.CDLL(dll_path)
        self._setup_functions()

    def _setup_functions(self):
        self._dll.process_batch.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.uint64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.uint64, flags='C_CONTIGUOUS'),
            ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
        ]
        self._dll.process_batch.restype = ctypes.c_int

    def process_batch(self, items: list) -> np.ndarray:
        """여러 아이템을 단일 DLL 호출로 처리"""
        # 데이터 연결
        all_data = np.concatenate([np.asarray(item, dtype=np.float32) for item in items])

        # 오프셋과 길이 계산
        lengths = np.array([len(item) for item in items], dtype=np.uint64)
        offsets = np.zeros(len(items), dtype=np.uint64)
        offsets[1:] = np.cumsum(lengths[:-1])

        # 결과 버퍼
        results = np.zeros(len(items), dtype=np.float32)

        # 단일 호출 (GIL 한 번만 해제)
        self._dll.process_batch(all_data, offsets, lengths, len(items), results)

        return results
```

### GIL 해제 확인 방법

```python
import threading
import time
import ctypes

def test_gil_release(dll):
    """GIL이 실제로 해제되는지 테스트"""
    results = []

    def worker(worker_id, data):
        start = time.time()
        dll.long_operation(data, len(data))  # 1초 이상 걸리는 연산
        elapsed = time.time() - start
        results.append((worker_id, elapsed))

    # 4개 스레드 동시 시작
    threads = []
    for i in range(4):
        data = np.random.rand(1000000).astype(np.float32)
        t = threading.Thread(target=worker, args=(i, data))
        threads.append(t)

    start_all = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_time = time.time() - start_all

    # 결과 분석
    individual_times = [r[1] for r in results]
    avg_time = sum(individual_times) / len(individual_times)

    print(f"Total wall time: {total_time:.2f}s")
    print(f"Average individual time: {avg_time:.2f}s")
    print(f"Expected if GIL released: ~{avg_time:.2f}s")
    print(f"Expected if GIL held: ~{avg_time * 4:.2f}s")

    # GIL이 해제되면 total_time ≈ avg_time
    # GIL이 유지되면 total_time ≈ avg_time * 4
    if total_time < avg_time * 2:
        print("✓ GIL appears to be released during DLL calls")
    else:
        print("✗ GIL appears to be held during DLL calls")
```

### 주의사항

1. **콜백 함수**: Python 콜백을 DLL에 전달하면 콜백 실행 시 GIL 재획득 필요
2. **Python 객체 접근**: GIL 해제 중 Python 객체 수정 금지
3. **스레드 안전성**: DLL 내부가 스레드-안전해야 함
4. **예외 처리**: GIL 해제 중 예외 발생 시 주의 필요

```python
# 콜백과 GIL
def callback_with_gil(progress):
    """이 콜백은 GIL을 재획득함"""
    print(f"Progress: {progress}%")  # Python 코드 실행

# DLL에서 콜백 호출 시 성능 저하 가능
# 해결책: 콜백 빈도 줄이기 또는 progress를 공유 메모리로
```
