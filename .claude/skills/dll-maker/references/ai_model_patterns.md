# AI 모델 특화 패턴

## 모델 라이프사이클 관리

### init/cleanup 패턴

```cpp
// model_lifecycle.h
#pragma once

#ifdef DLL_EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif

extern "C" {
    // 모델 핸들 (불투명 포인터)
    typedef void* ModelHandle;

    /**
     * 모델 초기화
     * @param config_path 설정 파일 경로 (NULL이면 기본값 사용)
     * @return 모델 핸들, 실패 시 NULL
     */
    API ModelHandle model_init(const char* config_path);

    /**
     * 모델 정리 및 메모리 해제
     * @param handle 모델 핸들
     */
    API void model_cleanup(ModelHandle handle);

    /**
     * 모델이 유효한지 확인
     * @param handle 모델 핸들
     * @return 1: 유효, 0: 무효
     */
    API int model_is_valid(ModelHandle handle);

    /**
     * 모델 버전 정보
     * @return 버전 문자열 (예: "1.0.0")
     */
    API const char* model_version(void);
}
```

```cpp
// model_lifecycle.cpp
#define DLL_EXPORT
#include "model_lifecycle.h"

#include <memory>
#include <mutex>
#include <unordered_set>

// 내부 모델 클래스
class Model {
public:
    float* weights = nullptr;
    size_t weight_count = 0;
    bool initialized = false;

    Model() = default;
    ~Model() {
        if (weights) {
            delete[] weights;
            weights = nullptr;
        }
    }

    bool load(const char* config_path) {
        // 가중치 로드 로직
        initialized = true;
        return true;
    }
};

// 핸들 추적 (메모리 누수 방지)
static std::mutex g_handles_mutex;
static std::unordered_set<Model*> g_valid_handles;

static const char* VERSION = "1.0.0";

extern "C" {

API ModelHandle model_init(const char* config_path) {
    try {
        auto* model = new Model();
        if (!model->load(config_path)) {
            delete model;
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(g_handles_mutex);
        g_valid_handles.insert(model);

        return static_cast<ModelHandle>(model);
    } catch (...) {
        return nullptr;
    }
}

API void model_cleanup(ModelHandle handle) {
    if (!handle) return;

    std::lock_guard<std::mutex> lock(g_handles_mutex);
    auto* model = static_cast<Model*>(handle);

    if (g_valid_handles.count(model)) {
        g_valid_handles.erase(model);
        delete model;
    }
}

API int model_is_valid(ModelHandle handle) {
    if (!handle) return 0;

    std::lock_guard<std::mutex> lock(g_handles_mutex);
    return g_valid_handles.count(static_cast<Model*>(handle)) ? 1 : 0;
}

API const char* model_version(void) {
    return VERSION;
}

}
```

### Python 래퍼 (Context Manager 지원)

```python
import ctypes
from pathlib import Path
from typing import Optional

class AIModel:
    """AI 모델 DLL 래퍼 (Context Manager 지원)"""

    def __init__(self, dll_path: Optional[str] = None, config_path: Optional[str] = None):
        if dll_path is None:
            dll_path = Path(__file__).parent / "model.dll"
        self._dll = ctypes.CDLL(str(dll_path))
        self._setup_functions()

        # 모델 초기화
        config = config_path.encode('utf-8') if config_path else None
        self._handle = self._dll.model_init(config)
        if not self._handle:
            raise RuntimeError("Failed to initialize model")

    def _setup_functions(self):
        self._dll.model_init.argtypes = [ctypes.c_char_p]
        self._dll.model_init.restype = ctypes.c_void_p

        self._dll.model_cleanup.argtypes = [ctypes.c_void_p]
        self._dll.model_cleanup.restype = None

        self._dll.model_is_valid.argtypes = [ctypes.c_void_p]
        self._dll.model_is_valid.restype = ctypes.c_int

        self._dll.model_version.argtypes = []
        self._dll.model_version.restype = ctypes.c_char_p

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """명시적 정리"""
        if hasattr(self, '_handle') and self._handle:
            self._dll.model_cleanup(self._handle)
            self._handle = None

    @property
    def is_valid(self) -> bool:
        return bool(self._dll.model_is_valid(self._handle))

    @property
    def version(self) -> str:
        return self._dll.model_version().decode('utf-8')


# 사용 예시
if __name__ == "__main__":
    # Context manager 사용 (권장)
    with AIModel() as model:
        print(f"Model version: {model.version}")
        # 추론 수행...

    # 또는 명시적 관리
    model = AIModel()
    try:
        # 추론 수행...
        pass
    finally:
        model.cleanup()
```

---

## 가중치 파일 로딩

### 바이너리 포맷 (권장)

```cpp
// weights_loader.h
#pragma once

#include <cstdint>
#include <cstddef>

// 가중치 파일 헤더
#pragma pack(push, 1)
struct WeightFileHeader {
    char magic[4];           // "WGHT"
    uint32_t version;        // 파일 포맷 버전
    uint32_t num_layers;     // 레이어 수
    uint32_t total_params;   // 전체 파라미터 수
    uint32_t dtype;          // 0: float32, 1: float16, 2: int8
    uint32_t reserved[3];    // 예약
};

struct LayerInfo {
    char name[64];           // 레이어 이름
    uint32_t shape[4];       // 최대 4차원 shape
    uint32_t ndim;           // 실제 차원 수
    uint64_t offset;         // 데이터 오프셋
    uint64_t size_bytes;     // 데이터 크기
};
#pragma pack(pop)
```

```cpp
// weights_loader.cpp
#include "weights_loader.h"
#include <fstream>
#include <vector>
#include <cstring>

class WeightsLoader {
public:
    struct Layer {
        std::string name;
        std::vector<uint32_t> shape;
        std::vector<float> data;
    };

private:
    std::vector<Layer> layers_;
    bool loaded_ = false;

public:
    bool load(const char* filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) return false;

        // 헤더 읽기
        WeightFileHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (std::memcmp(header.magic, "WGHT", 4) != 0) {
            return false;  // 잘못된 파일 포맷
        }

        // 레이어 정보 읽기
        std::vector<LayerInfo> layer_infos(header.num_layers);
        file.read(reinterpret_cast<char*>(layer_infos.data()),
                  sizeof(LayerInfo) * header.num_layers);

        // 각 레이어 데이터 로드
        layers_.resize(header.num_layers);
        for (uint32_t i = 0; i < header.num_layers; i++) {
            const auto& info = layer_infos[i];
            auto& layer = layers_[i];

            layer.name = info.name;
            for (uint32_t d = 0; d < info.ndim; d++) {
                layer.shape.push_back(info.shape[d]);
            }

            size_t num_elements = info.size_bytes / sizeof(float);
            layer.data.resize(num_elements);

            file.seekg(info.offset);
            file.read(reinterpret_cast<char*>(layer.data.data()), info.size_bytes);
        }

        loaded_ = true;
        return true;
    }

    const Layer* get_layer(const char* name) const {
        for (const auto& layer : layers_) {
            if (layer.name == name) return &layer;
        }
        return nullptr;
    }

    bool is_loaded() const { return loaded_; }
};

// DLL 인터페이스
static WeightsLoader g_loader;

extern "C" {

API int load_weights(const char* filepath) {
    return g_loader.load(filepath) ? 0 : -1;
}

API int get_layer_data(const char* name, float* buffer, size_t buffer_size) {
    const auto* layer = g_loader.get_layer(name);
    if (!layer) return -1;

    size_t copy_size = std::min(buffer_size, layer->data.size());
    std::memcpy(buffer, layer->data.data(), copy_size * sizeof(float));

    return static_cast<int>(copy_size);
}

}
```

### Python에서 가중치 파일 생성

```python
import numpy as np
import struct
from pathlib import Path

def save_weights(layers: dict, filepath: str):
    """
    가중치를 바이너리 파일로 저장

    Args:
        layers: {"layer_name": np.ndarray, ...}
        filepath: 출력 파일 경로
    """
    with open(filepath, 'wb') as f:
        # 헤더
        f.write(b'WGHT')                              # magic
        f.write(struct.pack('I', 1))                  # version
        f.write(struct.pack('I', len(layers)))        # num_layers
        total_params = sum(arr.size for arr in layers.values())
        f.write(struct.pack('I', total_params))       # total_params
        f.write(struct.pack('I', 0))                  # dtype (float32)
        f.write(struct.pack('III', 0, 0, 0))          # reserved

        # 레이어 정보 (오프셋은 나중에 계산)
        layer_info_start = f.tell()
        layer_infos = []

        for name, arr in layers.items():
            arr = arr.astype(np.float32)
            shape = list(arr.shape) + [0] * (4 - len(arr.shape))

            info = {
                'name': name.encode('utf-8')[:63].ljust(64, b'\x00'),
                'shape': shape,
                'ndim': len(arr.shape),
                'data': arr.tobytes()
            }
            layer_infos.append(info)

            # placeholder 쓰기
            f.write(info['name'])
            f.write(struct.pack('IIII', *shape))
            f.write(struct.pack('I', info['ndim']))
            f.write(struct.pack('QQ', 0, 0))  # offset, size (나중에)

        # 데이터 쓰기 및 오프셋 업데이트
        data_start = f.tell()
        offsets = []

        for info in layer_infos:
            offsets.append(f.tell())
            f.write(info['data'])

        # 오프셋 업데이트
        for i, (info, offset) in enumerate(zip(layer_infos, offsets)):
            pos = layer_info_start + i * (64 + 16 + 4 + 16) + 64 + 16 + 4
            f.seek(pos)
            f.write(struct.pack('QQ', offset, len(info['data'])))


# 사용 예시
if __name__ == "__main__":
    # PyTorch 모델에서 가중치 추출
    # model = torch.load("model.pth")
    # layers = {name: param.numpy() for name, param in model.state_dict().items()}

    # 테스트용 더미 가중치
    layers = {
        "conv1.weight": np.random.randn(32, 3, 3, 3).astype(np.float32),
        "conv1.bias": np.random.randn(32).astype(np.float32),
        "fc.weight": np.random.randn(10, 512).astype(np.float32),
        "fc.bias": np.random.randn(10).astype(np.float32),
    }

    save_weights(layers, "model_weights.bin")
    print("Weights saved!")
```

---

## 배치 처리

### C++ 구현

```cpp
// batch_inference.h
#pragma once

#ifdef DLL_EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif

#include <cstddef>

extern "C" {
    /**
     * 배치 추론 수행
     *
     * @param images      이미지 배치 (N x H x W x C, row-major)
     * @param batch_size  배치 크기
     * @param height      이미지 높이
     * @param width       이미지 너비
     * @param channels    채널 수
     * @param outputs     출력 버퍼 (N x num_classes)
     * @param num_classes 클래스 수
     * @return 0: 성공, 음수: 에러
     */
    API int batch_predict(
        const float* images,
        size_t batch_size,
        size_t height,
        size_t width,
        size_t channels,
        float* outputs,
        size_t num_classes
    );

    /**
     * 최적 배치 크기 조회
     * GPU 메모리 또는 CPU 캐시에 따라 결정
     */
    API size_t get_optimal_batch_size(void);
}
```

```cpp
// batch_inference.cpp
#define DLL_EXPORT
#include "batch_inference.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>

// 단일 이미지 처리 (내부 함수)
static void process_single(
    const float* image,
    size_t pixels,
    size_t channels,
    float* output,
    size_t num_classes
) {
    // 간단한 특징 추출 + 분류 (실제로는 모델 추론)
    std::vector<float> features(128, 0.0f);

    // 특징 계산 (예시)
    for (size_t i = 0; i < std::min(pixels * channels, features.size()); i++) {
        features[i % 128] += image[i];
    }

    // 분류 (예시)
    for (size_t c = 0; c < num_classes; c++) {
        output[c] = 0.0f;
        for (size_t f = 0; f < features.size(); f++) {
            output[c] += features[f] * 0.01f;  // 실제로는 학습된 가중치
        }
    }

    // Softmax
    float max_val = *std::max_element(output, output + num_classes);
    float sum = 0.0f;
    for (size_t c = 0; c < num_classes; c++) {
        output[c] = std::exp(output[c] - max_val);
        sum += output[c];
    }
    for (size_t c = 0; c < num_classes; c++) {
        output[c] /= sum;
    }
}

extern "C" {

API int batch_predict(
    const float* images,
    size_t batch_size,
    size_t height,
    size_t width,
    size_t channels,
    float* outputs,
    size_t num_classes
) {
    if (!images || !outputs) return -1;
    if (batch_size == 0 || height == 0 || width == 0) return -2;

    const size_t pixels = height * width;
    const size_t image_size = pixels * channels;

    // 멀티스레드 처리
    const size_t num_threads = std::min(
        batch_size,
        static_cast<size_t>(std::thread::hardware_concurrency())
    );

    if (num_threads <= 1 || batch_size < 4) {
        // 단일 스레드
        for (size_t i = 0; i < batch_size; i++) {
            process_single(
                images + i * image_size,
                pixels,
                channels,
                outputs + i * num_classes,
                num_classes
            );
        }
    } else {
        // 멀티스레드
        std::vector<std::thread> threads;
        const size_t chunk_size = (batch_size + num_threads - 1) / num_threads;

        for (size_t t = 0; t < num_threads; t++) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, batch_size);

            if (start >= batch_size) break;

            threads.emplace_back([=]() {
                for (size_t i = start; i < end; i++) {
                    process_single(
                        images + i * image_size,
                        pixels,
                        channels,
                        outputs + i * num_classes,
                        num_classes
                    );
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    return 0;
}

API size_t get_optimal_batch_size(void) {
    // CPU: L3 캐시 기준 (대략적)
    // GPU: VRAM 기준으로 계산
    return 32;  // 기본값
}

}
```

### Python 래퍼

```python
import ctypes
import numpy as np
from typing import List, Tuple

class BatchPredictor:
    def __init__(self, dll_path: str):
        self._dll = ctypes.CDLL(dll_path)
        self._setup_functions()
        self._optimal_batch = self._dll.get_optimal_batch_size()

    def _setup_functions(self):
        self._dll.batch_predict.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t,  # batch_size
            ctypes.c_size_t,  # height
            ctypes.c_size_t,  # width
            ctypes.c_size_t,  # channels
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t   # num_classes
        ]
        self._dll.batch_predict.restype = ctypes.c_int

        self._dll.get_optimal_batch_size.argtypes = []
        self._dll.get_optimal_batch_size.restype = ctypes.c_size_t

    def predict_batch(
        self,
        images: np.ndarray,
        num_classes: int = 10
    ) -> np.ndarray:
        """
        배치 추론 수행

        Args:
            images: (N, H, W, C) 또는 (N, H, W) 형태의 이미지 배열
            num_classes: 출력 클래스 수

        Returns:
            (N, num_classes) 형태의 확률 배열
        """
        images = np.ascontiguousarray(images, dtype=np.float32)

        if images.ndim == 3:
            # (N, H, W) -> (N, H, W, 1)
            images = images[..., np.newaxis]

        batch_size, height, width, channels = images.shape
        outputs = np.zeros((batch_size, num_classes), dtype=np.float32)

        ret = self._dll.batch_predict(
            images, batch_size, height, width, channels,
            outputs, num_classes
        )

        if ret != 0:
            raise RuntimeError(f"Batch prediction failed: {ret}")

        return outputs

    def predict_large_batch(
        self,
        images: np.ndarray,
        num_classes: int = 10
    ) -> np.ndarray:
        """
        큰 배치를 최적 크기로 분할하여 처리

        메모리 효율적인 처리를 위해 내부적으로 배치를 분할
        """
        total = len(images)
        results = []

        for i in range(0, total, self._optimal_batch):
            batch = images[i:i + self._optimal_batch]
            results.append(self.predict_batch(batch, num_classes))

        return np.vstack(results)


# 사용 예시
if __name__ == "__main__":
    predictor = BatchPredictor("model.dll")

    # 배치 이미지 생성
    images = np.random.rand(100, 224, 224, 3).astype(np.float32)

    # 배치 추론
    probs = predictor.predict_large_batch(images, num_classes=10)
    predictions = np.argmax(probs, axis=1)

    print(f"Predictions shape: {probs.shape}")
    print(f"First 5 predictions: {predictions[:5]}")
```

---

## GPU 지원 (CUDA)

### CUDA 커널 예시

```cpp
// gpu_inference.h
#pragma once

#ifdef DLL_EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif

extern "C" {
    // GPU 사용 가능 여부 확인
    API int cuda_is_available(void);

    // GPU 정보 조회
    API int cuda_get_device_count(void);
    API int cuda_get_device_name(int device_id, char* buffer, size_t buffer_size);

    // GPU 메모리 관리
    API void* cuda_malloc(size_t size);
    API void cuda_free(void* ptr);
    API int cuda_memcpy_to_device(void* dst, const void* src, size_t size);
    API int cuda_memcpy_to_host(void* dst, const void* src, size_t size);

    // GPU 추론
    API int cuda_batch_predict(
        const float* d_images,    // GPU 메모리의 이미지
        size_t batch_size,
        size_t height,
        size_t width,
        size_t channels,
        float* d_outputs,         // GPU 메모리의 출력
        size_t num_classes
    );
}
```

```cpp
// gpu_inference.cu (CUDA 파일)
#include "gpu_inference.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA 에러 체크 매크로
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) return -1; \
    } while(0)

// 간단한 추론 커널
__global__ void inference_kernel(
    const float* images,
    size_t image_size,
    float* outputs,
    size_t num_classes
) {
    int batch_idx = blockIdx.x;
    int class_idx = threadIdx.x;

    if (class_idx >= num_classes) return;

    const float* image = images + batch_idx * image_size;
    float* output = outputs + batch_idx * num_classes;

    // 간단한 연산 (실제로는 복잡한 네트워크)
    float sum = 0.0f;
    for (int i = class_idx; i < image_size; i += num_classes) {
        sum += image[i];
    }
    output[class_idx] = sum / image_size;
}

// Softmax 커널
__global__ void softmax_kernel(float* data, size_t batch_size, size_t num_classes) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    float* row = data + batch_idx * num_classes;

    // Max 찾기
    float max_val = row[0];
    for (int i = 1; i < num_classes; i++) {
        max_val = fmaxf(max_val, row[i]);
    }

    // Exp 및 합계
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        row[i] = expf(row[i] - max_val);
        sum += row[i];
    }

    // 정규화
    for (int i = 0; i < num_classes; i++) {
        row[i] /= sum;
    }
}

extern "C" {

API int cuda_is_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

API int cuda_get_device_count(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

API int cuda_get_device_name(int device_id, char* buffer, size_t buffer_size) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    strncpy(buffer, prop.name, buffer_size - 1);
    buffer[buffer_size - 1] = '\0';
    return 0;
}

API void* cuda_malloc(size_t size) {
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, size) != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

API void cuda_free(void* ptr) {
    if (ptr) cudaFree(ptr);
}

API int cuda_memcpy_to_device(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return 0;
}

API int cuda_memcpy_to_host(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return 0;
}

API int cuda_batch_predict(
    const float* d_images,
    size_t batch_size,
    size_t height,
    size_t width,
    size_t channels,
    float* d_outputs,
    size_t num_classes
) {
    size_t image_size = height * width * channels;

    // 추론 커널 실행
    inference_kernel<<<batch_size, num_classes>>>(
        d_images, image_size, d_outputs, num_classes
    );
    CUDA_CHECK(cudaGetLastError());

    // Softmax 커널 실행
    softmax_kernel<<<batch_size, 1>>>(d_outputs, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

}
```

### CMakeLists.txt (CUDA 지원)

```cmake
cmake_minimum_required(VERSION 3.18)
project(GPUModel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA 아키텍처 설정
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)  # Turing, Ampere

add_library(gpu_model SHARED
    src/gpu_inference.cu
)

target_include_directories(gpu_model PUBLIC include)

# CUDA 런타임 링크
target_link_libraries(gpu_model PRIVATE
    ${CUDA_LIBRARIES}
    cudart
)
```

### Python GPU 래퍼

```python
import ctypes
import numpy as np
from typing import Optional

class GPUModel:
    """GPU 가속 모델 래퍼"""

    def __init__(self, dll_path: str):
        self._dll = ctypes.CDLL(dll_path)
        self._setup_functions()

        if not self.is_cuda_available:
            raise RuntimeError("CUDA is not available")

        self._d_images = None
        self._d_outputs = None

    def _setup_functions(self):
        self._dll.cuda_is_available.restype = ctypes.c_int
        self._dll.cuda_get_device_count.restype = ctypes.c_int

        self._dll.cuda_malloc.argtypes = [ctypes.c_size_t]
        self._dll.cuda_malloc.restype = ctypes.c_void_p

        self._dll.cuda_free.argtypes = [ctypes.c_void_p]

        self._dll.cuda_memcpy_to_device.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
        ]
        self._dll.cuda_memcpy_to_host.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
        ]

    @property
    def is_cuda_available(self) -> bool:
        return bool(self._dll.cuda_is_available())

    @property
    def device_count(self) -> int:
        return self._dll.cuda_get_device_count()

    def predict_batch(
        self,
        images: np.ndarray,
        num_classes: int = 10
    ) -> np.ndarray:
        """GPU에서 배치 추론"""
        images = np.ascontiguousarray(images, dtype=np.float32)
        batch_size, height, width, channels = images.shape

        image_bytes = images.nbytes
        output_bytes = batch_size * num_classes * 4  # float32

        # GPU 메모리 할당
        d_images = self._dll.cuda_malloc(image_bytes)
        d_outputs = self._dll.cuda_malloc(output_bytes)

        try:
            # 입력 데이터 복사
            self._dll.cuda_memcpy_to_device(
                d_images,
                images.ctypes.data_as(ctypes.c_void_p),
                image_bytes
            )

            # 추론
            ret = self._dll.cuda_batch_predict(
                d_images, batch_size, height, width, channels,
                d_outputs, num_classes
            )
            if ret != 0:
                raise RuntimeError(f"CUDA inference failed: {ret}")

            # 결과 복사
            outputs = np.zeros((batch_size, num_classes), dtype=np.float32)
            self._dll.cuda_memcpy_to_host(
                outputs.ctypes.data_as(ctypes.c_void_p),
                d_outputs,
                output_bytes
            )

            return outputs

        finally:
            # GPU 메모리 해제
            self._dll.cuda_free(d_images)
            self._dll.cuda_free(d_outputs)
```

---

## ONNX Runtime 연동

### C++ ONNX 추론

```cpp
// onnx_inference.h
#pragma once

#ifdef DLL_EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif

extern "C" {
    typedef void* OnnxSession;

    // 세션 관리
    API OnnxSession onnx_create_session(const char* model_path, int use_gpu);
    API void onnx_destroy_session(OnnxSession session);

    // 추론
    API int onnx_run(
        OnnxSession session,
        const char* input_name,
        const float* input_data,
        const int64_t* input_shape,
        size_t input_dims,
        const char* output_name,
        float* output_data,
        size_t output_size
    );

    // 모델 정보
    API int onnx_get_input_count(OnnxSession session);
    API int onnx_get_output_count(OnnxSession session);
}
```

```cpp
// onnx_inference.cpp
#define DLL_EXPORT
#include "onnx_inference.h"

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

struct OnnxSessionWrapper {
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;

    OnnxSessionWrapper(const char* model_path, bool use_gpu)
        : env(ORT_LOGGING_LEVEL_WARNING, "DLL")
        , session(nullptr)
    {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(4);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (use_gpu) {
            // CUDA 프로바이더 추가
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            options.AppendExecutionProvider_CUDA(cuda_options);
        }

        session = Ort::Session(env, model_path, options);
    }
};

extern "C" {

API OnnxSession onnx_create_session(const char* model_path, int use_gpu) {
    try {
        auto* wrapper = new OnnxSessionWrapper(model_path, use_gpu != 0);
        return static_cast<OnnxSession>(wrapper);
    } catch (...) {
        return nullptr;
    }
}

API void onnx_destroy_session(OnnxSession session) {
    if (session) {
        delete static_cast<OnnxSessionWrapper*>(session);
    }
}

API int onnx_run(
    OnnxSession session,
    const char* input_name,
    const float* input_data,
    const int64_t* input_shape,
    size_t input_dims,
    const char* output_name,
    float* output_data,
    size_t output_size
) {
    if (!session) return -1;

    try {
        auto* wrapper = static_cast<OnnxSessionWrapper*>(session);

        // 입력 텐서 생성
        std::vector<int64_t> shape(input_shape, input_shape + input_dims);
        size_t input_size = 1;
        for (auto d : shape) input_size *= d;

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault
        );

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(input_data), input_size,
            shape.data(), shape.size()
        );

        // 추론 실행
        const char* input_names[] = {input_name};
        const char* output_names[] = {output_name};

        auto outputs = wrapper->session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1
        );

        // 결과 복사
        float* output_ptr = outputs[0].GetTensorMutableData<float>();
        auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
        size_t actual_size = output_info.GetElementCount();

        size_t copy_size = std::min(output_size, actual_size);
        std::memcpy(output_data, output_ptr, copy_size * sizeof(float));

        return static_cast<int>(copy_size);

    } catch (...) {
        return -1;
    }
}

API int onnx_get_input_count(OnnxSession session) {
    if (!session) return -1;
    auto* wrapper = static_cast<OnnxSessionWrapper*>(session);
    return static_cast<int>(wrapper->session.GetInputCount());
}

API int onnx_get_output_count(OnnxSession session) {
    if (!session) return -1;
    auto* wrapper = static_cast<OnnxSessionWrapper*>(session);
    return static_cast<int>(wrapper->session.GetOutputCount());
}

}
```

### CMakeLists.txt (ONNX Runtime)

```cmake
cmake_minimum_required(VERSION 3.14)
project(OnnxModel)

set(CMAKE_CXX_STANDARD 17)

# ONNX Runtime 경로 설정
set(ONNXRUNTIME_DIR "C:/onnxruntime" CACHE PATH "ONNX Runtime installation path")

add_library(onnx_model SHARED
    src/onnx_inference.cpp
)

target_include_directories(onnx_model PRIVATE
    ${ONNXRUNTIME_DIR}/include
)

target_link_directories(onnx_model PRIVATE
    ${ONNXRUNTIME_DIR}/lib
)

target_link_libraries(onnx_model PRIVATE
    onnxruntime
)

# DLL 복사
add_custom_command(TARGET onnx_model POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
    $<TARGET_FILE_DIR:onnx_model>
)
```

### Python ONNX 래퍼

```python
import ctypes
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

class OnnxDLL:
    """ONNX Runtime DLL 래퍼"""

    def __init__(self, dll_path: str, model_path: str, use_gpu: bool = False):
        self._dll = ctypes.CDLL(dll_path)
        self._setup_functions()

        # 세션 생성
        self._session = self._dll.onnx_create_session(
            model_path.encode('utf-8'),
            1 if use_gpu else 0
        )
        if not self._session:
            raise RuntimeError(f"Failed to load ONNX model: {model_path}")

    def _setup_functions(self):
        self._dll.onnx_create_session.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._dll.onnx_create_session.restype = ctypes.c_void_p

        self._dll.onnx_destroy_session.argtypes = [ctypes.c_void_p]

        self._dll.onnx_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.int64),
            ctypes.c_size_t,
            ctypes.c_char_p,
            np.ctypeslib.ndpointer(dtype=np.float32),
            ctypes.c_size_t
        ]
        self._dll.onnx_run.restype = ctypes.c_int

    def __del__(self):
        if hasattr(self, '_session') and self._session:
            self._dll.onnx_destroy_session(self._session)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._session:
            self._dll.onnx_destroy_session(self._session)
            self._session = None

    def run(
        self,
        input_data: np.ndarray,
        input_name: str = "input",
        output_name: str = "output",
        output_shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """
        ONNX 모델 추론

        Args:
            input_data: 입력 텐서
            input_name: 입력 노드 이름
            output_name: 출력 노드 이름
            output_shape: 출력 shape (None이면 자동 추정)
        """
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)
        input_shape = np.array(input_data.shape, dtype=np.int64)

        # 출력 버퍼 (충분히 크게)
        if output_shape:
            output_size = int(np.prod(output_shape))
        else:
            output_size = input_data.size  # 기본값

        output_data = np.zeros(output_size, dtype=np.float32)

        ret = self._dll.onnx_run(
            self._session,
            input_name.encode('utf-8'),
            input_data,
            input_shape,
            len(input_shape),
            output_name.encode('utf-8'),
            output_data,
            output_size
        )

        if ret < 0:
            raise RuntimeError(f"ONNX inference failed: {ret}")

        if output_shape:
            return output_data[:int(np.prod(output_shape))].reshape(output_shape)
        return output_data[:ret]


# 사용 예시
if __name__ == "__main__":
    # PyTorch 모델을 ONNX로 변환하는 코드
    """
    import torch
    model = MyModel()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    """

    # DLL 사용
    with OnnxDLL("onnx_model.dll", "model.onnx", use_gpu=True) as model:
        input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
        output = model.run(input_data, output_shape=(1, 1000))
        print(f"Output shape: {output.shape}")
        print(f"Top-5 classes: {np.argsort(output[0])[-5:][::-1]}")
```

---

## TensorRT 연동

NVIDIA TensorRT는 ONNX Runtime보다 더 높은 GPU 성능을 제공합니다. 특히 NVIDIA GPU에서 추론 최적화가 필요할 때 사용합니다.

### ONNX → TensorRT 엔진 변환

```cpp
// trt_builder.h
#pragma once

#ifdef DLL_EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif

extern "C" {
    /**
     * ONNX 모델을 TensorRT 엔진으로 변환
     *
     * @param onnx_path ONNX 모델 경로
     * @param engine_path 출력 엔진 파일 경로
     * @param fp16 FP16 모드 사용 여부
     * @param max_batch_size 최대 배치 크기
     * @return 0: 성공, 음수: 에러
     */
    API int trt_build_engine(
        const char* onnx_path,
        const char* engine_path,
        int fp16,
        int max_batch_size
    );
}
```

```cpp
// trt_builder.cpp
#define DLL_EXPORT
#include "trt_builder.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <memory>

using namespace nvinfer1;

// TensorRT 로거
class TRTLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            printf("[TRT] %s\n", msg);
        }
    }
};

static TRTLogger g_logger;

extern "C" {

API int trt_build_engine(
    const char* onnx_path,
    const char* engine_path,
    int fp16,
    int max_batch_size
) {
    // 빌더 생성
    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(g_logger));
    if (!builder) return -1;

    // 네트워크 정의 (explicit batch)
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
    );
    auto network = std::unique_ptr<INetworkDefinition>(
        builder->createNetworkV2(explicitBatch)
    );
    if (!network) return -2;

    // ONNX 파서
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, g_logger)
    );
    if (!parser->parseFromFile(onnx_path, static_cast<int>(ILogger::Severity::kWARNING))) {
        return -3;
    }

    // 빌드 설정
    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());

    // 메모리 풀 제한 (4GB)
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 4ULL << 30);

    // FP16 모드
    if (fp16 && builder->platformHasFastFp16()) {
        config->setFlag(BuilderFlag::kFP16);
    }

    // 동적 배치 크기 설정
    auto profile = builder->createOptimizationProfile();
    auto input = network->getInput(0);
    auto input_dims = input->getDimensions();

    // 최소/최적/최대 shape 설정
    Dims4 min_dims(1, input_dims.d[1], input_dims.d[2], input_dims.d[3]);
    Dims4 opt_dims(max_batch_size / 2, input_dims.d[1], input_dims.d[2], input_dims.d[3]);
    Dims4 max_dims(max_batch_size, input_dims.d[1], input_dims.d[2], input_dims.d[3]);

    profile->setDimensions(input->getName(), OptProfileSelector::kMIN, min_dims);
    profile->setDimensions(input->getName(), OptProfileSelector::kOPT, opt_dims);
    profile->setDimensions(input->getName(), OptProfileSelector::kMAX, max_dims);
    config->addOptimizationProfile(profile);

    // 엔진 빌드
    auto serialized = std::unique_ptr<IHostMemory>(
        builder->buildSerializedNetwork(*network, *config)
    );
    if (!serialized) return -4;

    // 파일로 저장
    std::ofstream file(engine_path, std::ios::binary);
    if (!file) return -5;
    file.write(static_cast<const char*>(serialized->data()), serialized->size());

    return 0;
}

}
```

### TensorRT 추론 엔진

```cpp
// trt_inference.h
#pragma once

#ifdef DLL_EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif

#include <cstddef>

extern "C" {
    typedef void* TRTEngine;

    // 엔진 로드/해제
    API TRTEngine trt_load_engine(const char* engine_path);
    API void trt_destroy_engine(TRTEngine engine);

    // 추론
    API int trt_infer(
        TRTEngine engine,
        const float* input,
        size_t batch_size,
        float* output,
        size_t output_size
    );

    // 비동기 추론 (CUDA 스트림 사용)
    API int trt_infer_async(
        TRTEngine engine,
        const float* input,
        size_t batch_size,
        float* output,
        size_t output_size,
        void* stream
    );

    // 정보 조회
    API int trt_get_input_dims(TRTEngine engine, int* dims, size_t max_dims);
    API int trt_get_output_dims(TRTEngine engine, int* dims, size_t max_dims);
    API size_t trt_get_max_batch_size(TRTEngine engine);
}
```

```cpp
// trt_inference.cpp
#define DLL_EXPORT
#include "trt_inference.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <memory>

using namespace nvinfer1;

class TRTLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            printf("[TRT] %s\n", msg);
        }
    }
};

struct TRTEngineWrapper {
    std::unique_ptr<IRuntime> runtime;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;

    // GPU 버퍼
    void* d_input = nullptr;
    void* d_output = nullptr;
    size_t input_size = 0;
    size_t output_size = 0;

    // 바인딩 정보
    int input_index = -1;
    int output_index = -1;
    Dims input_dims;
    Dims output_dims;

    ~TRTEngineWrapper() {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }
};

static TRTLogger g_logger;

extern "C" {

API TRTEngine trt_load_engine(const char* engine_path) {
    // 엔진 파일 읽기
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) return nullptr;

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) return nullptr;

    // 래퍼 생성
    auto* wrapper = new TRTEngineWrapper();

    // 런타임 및 엔진 생성
    wrapper->runtime.reset(createInferRuntime(g_logger));
    if (!wrapper->runtime) {
        delete wrapper;
        return nullptr;
    }

    wrapper->engine.reset(
        wrapper->runtime->deserializeCudaEngine(buffer.data(), size)
    );
    if (!wrapper->engine) {
        delete wrapper;
        return nullptr;
    }

    // 실행 컨텍스트 생성
    wrapper->context.reset(wrapper->engine->createExecutionContext());
    if (!wrapper->context) {
        delete wrapper;
        return nullptr;
    }

    // 바인딩 인덱스 찾기
    for (int i = 0; i < wrapper->engine->getNbIOTensors(); i++) {
        const char* name = wrapper->engine->getIOTensorName(i);
        if (wrapper->engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            wrapper->input_index = i;
            wrapper->input_dims = wrapper->engine->getTensorShape(name);
        } else {
            wrapper->output_index = i;
            wrapper->output_dims = wrapper->engine->getTensorShape(name);
        }
    }

    // 버퍼 크기 계산 (최대 배치 기준)
    wrapper->input_size = 1;
    for (int i = 0; i < wrapper->input_dims.nbDims; i++) {
        wrapper->input_size *= abs(wrapper->input_dims.d[i]);
    }

    wrapper->output_size = 1;
    for (int i = 0; i < wrapper->output_dims.nbDims; i++) {
        wrapper->output_size *= abs(wrapper->output_dims.d[i]);
    }

    // GPU 메모리 할당
    cudaMalloc(&wrapper->d_input, wrapper->input_size * sizeof(float));
    cudaMalloc(&wrapper->d_output, wrapper->output_size * sizeof(float));

    return static_cast<TRTEngine>(wrapper);
}

API void trt_destroy_engine(TRTEngine engine) {
    if (engine) {
        delete static_cast<TRTEngineWrapper*>(engine);
    }
}

API int trt_infer(
    TRTEngine engine,
    const float* input,
    size_t batch_size,
    float* output,
    size_t output_size
) {
    if (!engine || !input || !output) return -1;

    auto* wrapper = static_cast<TRTEngineWrapper*>(engine);

    // 입력 shape 설정 (동적 배치)
    Dims4 input_shape(
        batch_size,
        wrapper->input_dims.d[1],
        wrapper->input_dims.d[2],
        wrapper->input_dims.d[3]
    );

    const char* input_name = wrapper->engine->getIOTensorName(wrapper->input_index);
    wrapper->context->setInputShape(input_name, input_shape);

    // 입력 크기 계산
    size_t input_elements = batch_size;
    for (int i = 1; i < wrapper->input_dims.nbDims; i++) {
        input_elements *= wrapper->input_dims.d[i];
    }

    // 입력 데이터 복사 (Host → Device)
    cudaMemcpy(wrapper->d_input, input,
               input_elements * sizeof(float), cudaMemcpyHostToDevice);

    // 텐서 주소 설정
    const char* output_name = wrapper->engine->getIOTensorName(wrapper->output_index);
    wrapper->context->setTensorAddress(input_name, wrapper->d_input);
    wrapper->context->setTensorAddress(output_name, wrapper->d_output);

    // 추론 실행
    if (!wrapper->context->enqueueV3(nullptr)) {
        return -2;
    }
    cudaDeviceSynchronize();

    // 출력 크기 계산
    auto out_shape = wrapper->context->getTensorShape(output_name);
    size_t output_elements = 1;
    for (int i = 0; i < out_shape.nbDims; i++) {
        output_elements *= out_shape.d[i];
    }

    // 출력 데이터 복사 (Device → Host)
    size_t copy_size = std::min(output_size, output_elements);
    cudaMemcpy(output, wrapper->d_output,
               copy_size * sizeof(float), cudaMemcpyDeviceToHost);

    return static_cast<int>(copy_size);
}

API int trt_infer_async(
    TRTEngine engine,
    const float* input,
    size_t batch_size,
    float* output,
    size_t output_size,
    void* stream
) {
    if (!engine || !input || !output) return -1;

    auto* wrapper = static_cast<TRTEngineWrapper*>(engine);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // 입력 shape 설정
    Dims4 input_shape(
        batch_size,
        wrapper->input_dims.d[1],
        wrapper->input_dims.d[2],
        wrapper->input_dims.d[3]
    );

    const char* input_name = wrapper->engine->getIOTensorName(wrapper->input_index);
    const char* output_name = wrapper->engine->getIOTensorName(wrapper->output_index);

    wrapper->context->setInputShape(input_name, input_shape);

    // 입력 크기 계산
    size_t input_elements = batch_size;
    for (int i = 1; i < wrapper->input_dims.nbDims; i++) {
        input_elements *= wrapper->input_dims.d[i];
    }

    // 비동기 복사 및 실행
    cudaMemcpyAsync(wrapper->d_input, input,
                    input_elements * sizeof(float),
                    cudaMemcpyHostToDevice, cuda_stream);

    wrapper->context->setTensorAddress(input_name, wrapper->d_input);
    wrapper->context->setTensorAddress(output_name, wrapper->d_output);

    if (!wrapper->context->enqueueV3(cuda_stream)) {
        return -2;
    }

    // 출력 크기 계산
    auto out_shape = wrapper->context->getTensorShape(output_name);
    size_t output_elements = 1;
    for (int i = 0; i < out_shape.nbDims; i++) {
        output_elements *= out_shape.d[i];
    }

    size_t copy_size = std::min(output_size, output_elements);
    cudaMemcpyAsync(output, wrapper->d_output,
                    copy_size * sizeof(float),
                    cudaMemcpyDeviceToHost, cuda_stream);

    return static_cast<int>(copy_size);
}

API int trt_get_input_dims(TRTEngine engine, int* dims, size_t max_dims) {
    if (!engine || !dims) return -1;

    auto* wrapper = static_cast<TRTEngineWrapper*>(engine);
    int ndims = std::min(static_cast<size_t>(wrapper->input_dims.nbDims), max_dims);

    for (int i = 0; i < ndims; i++) {
        dims[i] = wrapper->input_dims.d[i];
    }

    return ndims;
}

API int trt_get_output_dims(TRTEngine engine, int* dims, size_t max_dims) {
    if (!engine || !dims) return -1;

    auto* wrapper = static_cast<TRTEngineWrapper*>(engine);
    int ndims = std::min(static_cast<size_t>(wrapper->output_dims.nbDims), max_dims);

    for (int i = 0; i < ndims; i++) {
        dims[i] = wrapper->output_dims.d[i];
    }

    return ndims;
}

API size_t trt_get_max_batch_size(TRTEngine engine) {
    if (!engine) return 0;
    auto* wrapper = static_cast<TRTEngineWrapper*>(engine);
    return abs(wrapper->input_dims.d[0]);
}

}
```

### CMakeLists.txt (TensorRT)

```cmake
cmake_minimum_required(VERSION 3.18)
project(TensorRTModel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# TensorRT 경로 (환경에 맞게 수정)
set(TENSORRT_DIR "C:/TensorRT-8.6" CACHE PATH "TensorRT installation path")

# CUDA
find_package(CUDA REQUIRED)

add_library(trt_model SHARED
    src/trt_builder.cpp
    src/trt_inference.cpp
)

target_include_directories(trt_model PRIVATE
    ${TENSORRT_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

target_link_directories(trt_model PRIVATE
    ${TENSORRT_DIR}/lib
)

target_link_libraries(trt_model PRIVATE
    nvinfer
    nvonnxparser
    ${CUDA_LIBRARIES}
    cudart
)

# DLL 복사
add_custom_command(TARGET trt_model POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${TENSORRT_DIR}/lib/nvinfer.dll"
    "${TENSORRT_DIR}/lib/nvonnxparser.dll"
    $<TARGET_FILE_DIR:trt_model>
)
```

### Python TensorRT 래퍼

```python
import ctypes
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

class TensorRTDLL:
    """TensorRT DLL 래퍼"""

    def __init__(self, dll_path: str, engine_path: str):
        """
        Args:
            dll_path: DLL 파일 경로
            engine_path: TensorRT 엔진 파일 경로 (.engine 또는 .plan)
        """
        self._dll = ctypes.CDLL(dll_path)
        self._setup_functions()

        # 엔진 로드
        self._engine = self._dll.trt_load_engine(engine_path.encode('utf-8'))
        if not self._engine:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        # 차원 정보 캐시
        self._input_dims = self._get_dims(is_input=True)
        self._output_dims = self._get_dims(is_input=False)

    def _setup_functions(self):
        self._dll.trt_load_engine.argtypes = [ctypes.c_char_p]
        self._dll.trt_load_engine.restype = ctypes.c_void_p

        self._dll.trt_destroy_engine.argtypes = [ctypes.c_void_p]

        self._dll.trt_infer.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t
        ]
        self._dll.trt_infer.restype = ctypes.c_int

        self._dll.trt_get_input_dims.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t
        ]
        self._dll.trt_get_input_dims.restype = ctypes.c_int

        self._dll.trt_get_output_dims.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t
        ]
        self._dll.trt_get_output_dims.restype = ctypes.c_int

    def _get_dims(self, is_input: bool) -> List[int]:
        dims = np.zeros(8, dtype=np.int32)
        func = self._dll.trt_get_input_dims if is_input else self._dll.trt_get_output_dims
        ndims = func(self._engine, dims, 8)
        return list(dims[:ndims])

    def __del__(self):
        if hasattr(self, '_engine') and self._engine:
            self._dll.trt_destroy_engine(self._engine)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._engine:
            self._dll.trt_destroy_engine(self._engine)
            self._engine = None

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """입력 shape (배치 차원은 -1로 표시)"""
        return tuple(self._input_dims)

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """출력 shape (배치 차원은 -1로 표시)"""
        return tuple(self._output_dims)

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        동기 추론

        Args:
            input_data: 입력 텐서 (batch, C, H, W)

        Returns:
            출력 텐서
        """
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)

        if input_data.ndim == 3:
            input_data = input_data[np.newaxis, ...]

        batch_size = input_data.shape[0]

        # 출력 버퍼 할당
        output_shape = list(self._output_dims)
        output_shape[0] = batch_size  # 배치 크기 설정
        output_size = int(np.prod(output_shape))
        output = np.zeros(output_size, dtype=np.float32)

        # 추론
        ret = self._dll.trt_infer(
            self._engine,
            input_data.flatten(),
            batch_size,
            output,
            output_size
        )

        if ret < 0:
            raise RuntimeError(f"TensorRT inference failed: {ret}")

        return output.reshape(output_shape)

    def benchmark(self, input_shape: Tuple[int, ...], iterations: int = 100) -> dict:
        """
        성능 벤치마크

        Args:
            input_shape: 입력 shape (batch, C, H, W)
            iterations: 반복 횟수

        Returns:
            벤치마크 결과 딕셔너리
        """
        import time

        input_data = np.random.rand(*input_shape).astype(np.float32)

        # 워밍업
        for _ in range(10):
            self.infer(input_data)

        # 벤치마크
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.infer(input_data)
            times.append(time.perf_counter() - start)

        times = np.array(times) * 1000  # ms

        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "throughput_fps": float(input_shape[0] * 1000 / np.mean(times))
        }


class TensorRTBuilder:
    """TensorRT 엔진 빌더"""

    def __init__(self, dll_path: str):
        self._dll = ctypes.CDLL(dll_path)

        self._dll.trt_build_engine.argtypes = [
            ctypes.c_char_p,  # onnx_path
            ctypes.c_char_p,  # engine_path
            ctypes.c_int,     # fp16
            ctypes.c_int      # max_batch_size
        ]
        self._dll.trt_build_engine.restype = ctypes.c_int

    def build(
        self,
        onnx_path: str,
        engine_path: str,
        fp16: bool = True,
        max_batch_size: int = 16
    ) -> bool:
        """
        ONNX 모델을 TensorRT 엔진으로 변환

        Args:
            onnx_path: ONNX 모델 경로
            engine_path: 출력 엔진 파일 경로
            fp16: FP16 모드 사용 여부
            max_batch_size: 최대 배치 크기

        Returns:
            성공 여부
        """
        ret = self._dll.trt_build_engine(
            onnx_path.encode('utf-8'),
            engine_path.encode('utf-8'),
            1 if fp16 else 0,
            max_batch_size
        )

        if ret != 0:
            errors = {
                -1: "Failed to create builder",
                -2: "Failed to create network",
                -3: "Failed to parse ONNX file",
                -4: "Failed to build engine",
                -5: "Failed to save engine file"
            }
            raise RuntimeError(errors.get(ret, f"Unknown error: {ret}"))

        return True


# 사용 예시
if __name__ == "__main__":
    dll_path = "trt_model.dll"

    # 1. ONNX → TensorRT 변환
    builder = TensorRTBuilder(dll_path)
    builder.build(
        onnx_path="model.onnx",
        engine_path="model.engine",
        fp16=True,
        max_batch_size=32
    )
    print("Engine built successfully!")

    # 2. 추론
    with TensorRTDLL(dll_path, "model.engine") as trt:
        print(f"Input shape: {trt.input_shape}")
        print(f"Output shape: {trt.output_shape}")

        # 단일 이미지
        input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
        output = trt.infer(input_data)
        print(f"Output: {output.shape}")

        # 배치 추론
        batch_input = np.random.rand(8, 3, 224, 224).astype(np.float32)
        batch_output = trt.infer(batch_input)
        print(f"Batch output: {batch_output.shape}")

        # 벤치마크
        results = trt.benchmark((16, 3, 224, 224), iterations=100)
        print(f"Throughput: {results['throughput_fps']:.1f} FPS")
        print(f"Latency: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
```

### ONNX Runtime vs TensorRT 비교

| 특성 | ONNX Runtime | TensorRT |
|------|-------------|----------|
| 플랫폼 | CPU, GPU (다양한 벤더) | NVIDIA GPU 전용 |
| 설치 | 간단 (pip install) | CUDA/cuDNN/TensorRT 설치 필요 |
| 성능 | 좋음 | 최고 (NVIDIA GPU) |
| 모델 변환 | 불필요 (ONNX 직접 사용) | 엔진 빌드 필요 (시간 소요) |
| 동적 shape | 쉬움 | 빌드 시 프로파일 설정 필요 |
| FP16/INT8 | 지원 | 최적화된 지원 |
| 메모리 | 동적 | 빌드 시 결정 |

### 권장 사용 시나리오

```python
def choose_runtime(requirements: dict) -> str:
    """
    요구사항에 따른 런타임 선택

    Args:
        requirements: {
            "gpu": "nvidia" | "amd" | "none",
            "latency_critical": bool,
            "dynamic_batch": bool,
            "deployment_simplicity": bool
        }
    """
    gpu = requirements.get("gpu", "none")
    latency_critical = requirements.get("latency_critical", False)
    dynamic_batch = requirements.get("dynamic_batch", True)
    simple_deploy = requirements.get("deployment_simplicity", True)

    if gpu == "nvidia" and latency_critical and not dynamic_batch:
        return "TensorRT"  # 최고 성능
    elif gpu != "none" and not simple_deploy:
        return "TensorRT" if gpu == "nvidia" else "ONNX Runtime"
    else:
        return "ONNX Runtime"  # 범용성, 간편함

# 예시
print(choose_runtime({
    "gpu": "nvidia",
    "latency_critical": True,
    "dynamic_batch": False
}))  # → TensorRT

print(choose_runtime({
    "gpu": "nvidia",
    "latency_critical": False,
    "deployment_simplicity": True
}))  # → ONNX Runtime
```
