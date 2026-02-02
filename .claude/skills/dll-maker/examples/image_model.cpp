#define DLL_EXPORT
#include "image_model.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <mutex>
#include <unordered_set>

// std::clamp 대체 (C++17 미지원 컴파일러용)
template<typename T>
inline T clamp_value(T val, T lo, T hi) {
    return (val < lo) ? lo : (val > hi) ? hi : val;
}

// ========================================
// 상수 정의
// ========================================
static const char* VERSION = "2.0.0";
constexpr int NUM_CLASSES = 10;
constexpr int NUM_FEATURES = 10;  // mean, std, 8 histogram bins
constexpr size_t OPTIMAL_BATCH_SIZE = 32;

// ========================================
// 내부 모델 클래스
// ========================================
class ImageModel {
public:
    float weights[NUM_FEATURES][NUM_CLASSES];
    bool initialized = false;

    ImageModel() {
        init_weights();
    }

    void init_weights() {
        if (initialized) return;

        // 간단한 pseudo-random (seed 42와 유사한 결과)
        unsigned int seed = 42;
        for (int i = 0; i < NUM_FEATURES; i++) {
            for (int j = 0; j < NUM_CLASSES; j++) {
                seed = seed * 1103515245 + 12345;
                float r = ((seed / 65536) % 32768) / 32768.0f;
                weights[i][j] = (r - 0.5f) * 2.0f;  // -1 ~ 1
            }
        }
        initialized = true;
    }
};

// ========================================
// 핸들 관리
// ========================================
static std::mutex g_handles_mutex;
static std::unordered_set<ImageModel*> g_valid_handles;

static bool is_valid_handle(ImageModelHandle handle) {
    if (!handle) return false;
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    return g_valid_handles.count(static_cast<ImageModel*>(handle)) > 0;
}

// ========================================
// 내부 헬퍼 함수
// ========================================
static void softmax(float* logits, int size) {
    float max_val = *std::max_element(logits, logits + size);

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }

    for (int i = 0; i < size; i++) {
        logits[i] /= sum;
    }
}

static int extract_features_internal(
    const float* image_data,
    size_t num_pixels,
    float* features
) {
    // 평균 계산
    double sum = 0.0;
    for (size_t i = 0; i < num_pixels; i++) {
        sum += image_data[i];
    }
    float mean_val = static_cast<float>(sum / num_pixels);

    // 표준편차 계산
    double var_sum = 0.0;
    for (size_t i = 0; i < num_pixels; i++) {
        double diff = image_data[i] - mean_val;
        var_sum += diff * diff;
    }
    float std_val = static_cast<float>(std::sqrt(var_sum / num_pixels));

    // 히스토그램 (8 bins, 0~255 범위)
    int hist[8] = {0};
    for (size_t i = 0; i < num_pixels; i++) {
        int bin = static_cast<int>(image_data[i] / 32.0f);
        bin = clamp_value(bin, 0, 7);
        hist[bin]++;
    }

    // 특징 벡터 구성
    features[0] = mean_val;
    features[1] = std_val;
    for (int i = 0; i < 8; i++) {
        features[2 + i] = static_cast<float>(hist[i]) / num_pixels;
    }

    return NUM_FEATURES;
}

static int predict_single(
    ImageModel* model,
    const float* image_data,
    size_t num_pixels,
    PredictionResult* result
) {
    // 특징 추출
    float features[NUM_FEATURES];
    extract_features_internal(image_data, num_pixels, features);

    // 분류 (선형 변환)
    float logits[NUM_CLASSES] = {0};
    for (int j = 0; j < NUM_CLASSES; j++) {
        for (int i = 0; i < NUM_FEATURES; i++) {
            logits[j] += features[i] * model->weights[i][j];
        }
    }

    // Softmax
    softmax(logits, NUM_CLASSES);

    // 최대값 찾기
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    result->class_index = max_idx;
    result->confidence = max_val;

    return IMG_SUCCESS;
}

// ========================================
// API 구현: 모델 라이프사이클
// ========================================
extern "C" {

API ImageModelHandle image_model_init(const char* config_path) {
    (void)config_path;  // 현재 미사용

    try {
        auto* model = new ImageModel();

        std::lock_guard<std::mutex> lock(g_handles_mutex);
        g_valid_handles.insert(model);

        return static_cast<ImageModelHandle>(model);
    } catch (...) {
        return nullptr;
    }
}

API void image_model_cleanup(ImageModelHandle handle) {
    if (!handle) return;

    std::lock_guard<std::mutex> lock(g_handles_mutex);
    auto* model = static_cast<ImageModel*>(handle);

    if (g_valid_handles.count(model)) {
        g_valid_handles.erase(model);
        delete model;
    }
}

API int image_model_is_valid(ImageModelHandle handle) {
    return is_valid_handle(handle) ? 1 : 0;
}

API const char* image_model_version(void) {
    return VERSION;
}

// ========================================
// API 구현: 단일 이미지 추론
// ========================================

API int predict_grayscale(
    ImageModelHandle handle,
    const float* image_data,
    size_t height,
    size_t width,
    PredictionResult* result
) {
    if (!is_valid_handle(handle)) return IMG_ERR_INVALID_HANDLE;
    if (!image_data || !result) return IMG_ERR_NULL_PTR;
    if (height == 0 || width == 0) return IMG_ERR_INVALID_SIZE;

    auto* model = static_cast<ImageModel*>(handle);
    size_t num_pixels = height * width;

    return predict_single(model, image_data, num_pixels, result);
}

API int predict_rgb(
    ImageModelHandle handle,
    const float* image_data,
    size_t height,
    size_t width,
    size_t channels,
    PredictionResult* result
) {
    if (!is_valid_handle(handle)) return IMG_ERR_INVALID_HANDLE;
    if (!image_data || !result) return IMG_ERR_NULL_PTR;
    if (height == 0 || width == 0 || channels == 0) return IMG_ERR_INVALID_SIZE;

    size_t num_pixels = height * width;

    // RGB to Grayscale 변환
    std::vector<float> grayscale(num_pixels);
    for (size_t i = 0; i < num_pixels; i++) {
        float sum = 0.0f;
        for (size_t c = 0; c < channels; c++) {
            sum += image_data[i * channels + c];
        }
        grayscale[i] = sum / channels;
    }

    auto* model = static_cast<ImageModel*>(handle);
    return predict_single(model, grayscale.data(), num_pixels, result);
}

// ========================================
// API 구현: 배치 처리
// ========================================

API int predict_batch(
    ImageModelHandle handle,
    const float* images,
    size_t batch_size,
    size_t height,
    size_t width,
    PredictionResult* results
) {
    if (!is_valid_handle(handle)) return IMG_ERR_INVALID_HANDLE;
    if (!images || !results) return IMG_ERR_NULL_PTR;
    if (batch_size == 0 || height == 0 || width == 0) return IMG_ERR_INVALID_SIZE;

    auto* model = static_cast<ImageModel*>(handle);
    size_t image_size = height * width;

    // 배치 처리 (순차 처리, 스레드 안전)
    for (size_t i = 0; i < batch_size; i++) {
        const float* image_data = images + i * image_size;
        int ret = predict_single(model, image_data, image_size, &results[i]);
        if (ret != IMG_SUCCESS) {
            return ret;
        }
    }

    return IMG_SUCCESS;
}

API size_t get_optimal_batch_size(void) {
    return OPTIMAL_BATCH_SIZE;
}

// ========================================
// API 구현: 특징 추출
// ========================================

API int extract_features(
    ImageModelHandle handle,
    const float* image_data,
    size_t num_pixels,
    float* features,
    size_t feature_len
) {
    if (!is_valid_handle(handle)) return IMG_ERR_INVALID_HANDLE;
    if (!image_data || !features) return IMG_ERR_NULL_PTR;
    if (num_pixels == 0 || feature_len < NUM_FEATURES) return IMG_ERR_INVALID_SIZE;

    return extract_features_internal(image_data, num_pixels, features);
}

}  // extern "C"
