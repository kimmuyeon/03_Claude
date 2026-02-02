#pragma once

#ifdef _WIN32
    #ifdef DLL_EXPORT
        #define API __declspec(dllexport)
    #else
        #define API __declspec(dllimport)
    #endif
#else
    #define API __attribute__((visibility("default")))
#endif

#include <cstdint>
#include <cstddef>

extern "C" {
    // ========================================
    // 모델 라이프사이클 API
    // ========================================

    /**
     * 모델 핸들 (불투명 포인터)
     */
    typedef void* ImageModelHandle;

    /**
     * 모델 초기화
     * @param config_path 설정 파일 경로 (NULL이면 기본값 사용)
     * @return 모델 핸들, 실패 시 NULL
     */
    API ImageModelHandle image_model_init(const char* config_path);

    /**
     * 모델 정리 및 메모리 해제
     * @param handle 모델 핸들
     */
    API void image_model_cleanup(ImageModelHandle handle);

    /**
     * 모델이 유효한지 확인
     * @param handle 모델 핸들
     * @return 1: 유효, 0: 무효
     */
    API int image_model_is_valid(ImageModelHandle handle);

    /**
     * 모델 버전 정보
     * @return 버전 문자열 (예: "1.0.0")
     */
    API const char* image_model_version(void);

    // ========================================
    // 데이터 구조체
    // ========================================

    /**
     * 이미지 분류 결과 구조체
     */
    #pragma pack(push, 1)
    struct PredictionResult {
        int class_index;      // 분류 클래스 인덱스
        float confidence;     // 신뢰도 (0.0 ~ 1.0)
    };
    #pragma pack(pop)

    // ========================================
    // 에러 코드
    // ========================================
    enum ImageModelError {
        IMG_SUCCESS = 0,
        IMG_ERR_NULL_PTR = -1,
        IMG_ERR_INVALID_SIZE = -2,
        IMG_ERR_INVALID_HANDLE = -3,
        IMG_ERR_OUT_OF_MEMORY = -4
    };

    // ========================================
    // 단일 이미지 추론 API
    // ========================================

    /**
     * 이미지 분류 수행 (Grayscale)
     *
     * @param handle      모델 핸들
     * @param image_data  이미지 픽셀 데이터 (float32, row-major)
     * @param height      이미지 높이
     * @param width       이미지 너비
     * @param result      결과 저장할 구조체 포인터
     * @return 0: 성공, 음수: 에러코드
     */
    API int predict_grayscale(
        ImageModelHandle handle,
        const float* image_data,
        size_t height,
        size_t width,
        PredictionResult* result
    );

    /**
     * 이미지 분류 수행 (RGB)
     *
     * @param handle      모델 핸들
     * @param image_data  이미지 픽셀 데이터 (float32, HWC 순서)
     * @param height      이미지 높이
     * @param width       이미지 너비
     * @param channels    채널 수 (3 for RGB)
     * @param result      결과 저장할 구조체 포인터
     * @return 0: 성공, 음수: 에러코드
     */
    API int predict_rgb(
        ImageModelHandle handle,
        const float* image_data,
        size_t height,
        size_t width,
        size_t channels,
        PredictionResult* result
    );

    // ========================================
    // 배치 처리 API
    // ========================================

    /**
     * 배치 이미지 분류 수행
     *
     * @param handle      모델 핸들
     * @param images      이미지 배치 (N x H x W, row-major, grayscale)
     * @param batch_size  배치 크기
     * @param height      이미지 높이
     * @param width       이미지 너비
     * @param results     결과 배열 (batch_size 크기)
     * @return 0: 성공, 음수: 에러코드
     */
    API int predict_batch(
        ImageModelHandle handle,
        const float* images,
        size_t batch_size,
        size_t height,
        size_t width,
        PredictionResult* results
    );

    /**
     * 최적 배치 크기 조회
     * @return 권장 배치 크기
     */
    API size_t get_optimal_batch_size(void);

    // ========================================
    // 특징 추출 API
    // ========================================

    /**
     * 특징 벡터 추출
     *
     * @param handle       모델 핸들
     * @param image_data   이미지 픽셀 데이터 (float32)
     * @param num_pixels   전체 픽셀 수
     * @param features     특징 벡터 출력 버퍼 (최소 10개 float)
     * @param feature_len  특징 버퍼 크기
     * @return 추출된 특징 수, 음수: 에러코드
     */
    API int extract_features(
        ImageModelHandle handle,
        const float* image_data,
        size_t num_pixels,
        float* features,
        size_t feature_len
    );
}
