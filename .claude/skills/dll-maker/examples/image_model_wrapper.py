"""
image_model.dll 용 Python ctypes 래퍼 (v2.0)

수정된 dll-maker 스킬의 패턴 적용:
- 모델 라이프사이클 (init/cleanup)
- Context Manager 지원
- 배치 처리
- 에러 처리 개선
- GIL 해제 지원
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List


# ========================================
# 에러 코드 및 예외
# ========================================
class ImageModelError(Exception):
    """DLL 에러 예외"""
    CODES = {
        0: "Success",
        -1: "Null pointer error",
        -2: "Invalid size error",
        -3: "Invalid handle error",
        -4: "Out of memory error"
    }

    def __init__(self, code: int):
        self.code = code
        message = self.CODES.get(code, f"Unknown error: {code}")
        super().__init__(message)


# ========================================
# 데이터 구조체
# ========================================
class PredictionResult(ctypes.Structure):
    """분류 결과 구조체"""
    _pack_ = 1
    _fields_ = [
        ("class_index", ctypes.c_int),
        ("confidence", ctypes.c_float)
    ]


# ========================================
# 메인 래퍼 클래스
# ========================================
class ImageModelDLL:
    """
    이미지 분류 DLL 래퍼 클래스 (v2.0)

    Context Manager 지원:
        with ImageModelDLL() as model:
            result = model.predict(image)

    수동 관리:
        model = ImageModelDLL()
        try:
            result = model.predict(image)
        finally:
            model.cleanup()
    """

    def __init__(self, dll_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Args:
            dll_path: DLL 파일 경로 (None이면 같은 폴더의 image_model.dll)
            config_path: 모델 설정 파일 경로 (None이면 기본값)
        """
        if dll_path is None:
            dll_path = Path(__file__).parent / "image_model.dll"
        self._dll = ctypes.CDLL(str(dll_path))
        self._setup_functions()

        # 모델 초기화
        config = config_path.encode('utf-8') if config_path else None
        self._handle = self._dll.image_model_init(config)
        if not self._handle:
            raise RuntimeError("Failed to initialize model")

    def _setup_functions(self):
        """DLL 함수 시그니처 설정"""
        # 라이프사이클 API
        self._dll.image_model_init.argtypes = [ctypes.c_char_p]
        self._dll.image_model_init.restype = ctypes.c_void_p

        self._dll.image_model_cleanup.argtypes = [ctypes.c_void_p]
        self._dll.image_model_cleanup.restype = None

        self._dll.image_model_is_valid.argtypes = [ctypes.c_void_p]
        self._dll.image_model_is_valid.restype = ctypes.c_int

        self._dll.image_model_version.argtypes = []
        self._dll.image_model_version.restype = ctypes.c_char_p

        # 단일 추론 API
        self._dll.predict_grayscale.argtypes = [
            ctypes.c_void_p,  # handle
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t,  # height
            ctypes.c_size_t,  # width
            ctypes.POINTER(PredictionResult)
        ]
        self._dll.predict_grayscale.restype = ctypes.c_int

        self._dll.predict_rgb.argtypes = [
            ctypes.c_void_p,  # handle
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t,  # height
            ctypes.c_size_t,  # width
            ctypes.c_size_t,  # channels
            ctypes.POINTER(PredictionResult)
        ]
        self._dll.predict_rgb.restype = ctypes.c_int

        # 배치 API
        self._dll.predict_batch.argtypes = [
            ctypes.c_void_p,  # handle
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t,  # batch_size
            ctypes.c_size_t,  # height
            ctypes.c_size_t,  # width
            ctypes.POINTER(PredictionResult)
        ]
        self._dll.predict_batch.restype = ctypes.c_int

        self._dll.get_optimal_batch_size.argtypes = []
        self._dll.get_optimal_batch_size.restype = ctypes.c_size_t

        # 특징 추출 API
        self._dll.extract_features.argtypes = [
            ctypes.c_void_p,  # handle
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t
        ]
        self._dll.extract_features.restype = ctypes.c_int

    # ========================================
    # Context Manager 지원
    # ========================================
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """명시적 리소스 정리"""
        if hasattr(self, '_handle') and self._handle:
            self._dll.image_model_cleanup(self._handle)
            self._handle = None

    # ========================================
    # 속성
    # ========================================
    @property
    def is_valid(self) -> bool:
        """모델이 유효한지 확인"""
        if not self._handle:
            return False
        return bool(self._dll.image_model_is_valid(self._handle))

    @property
    def version(self) -> str:
        """DLL 버전"""
        return self._dll.image_model_version().decode('utf-8')

    @property
    def optimal_batch_size(self) -> int:
        """권장 배치 크기"""
        return self._dll.get_optimal_batch_size()

    # ========================================
    # 단일 이미지 추론
    # ========================================
    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """
        이미지 분류 수행

        Args:
            image: numpy 배열 (H, W) 또는 (H, W, C)

        Returns:
            (class_index, confidence) 튜플

        Raises:
            ImageModelError: DLL 에러 발생 시
            ValueError: 잘못된 이미지 shape
        """
        if not self._handle:
            raise ImageModelError(-3)

        image = np.ascontiguousarray(image, dtype=np.float32)
        result = PredictionResult()

        if len(image.shape) == 2:
            h, w = image.shape
            ret = self._dll.predict_grayscale(
                self._handle, image, h, w, ctypes.byref(result)
            )
        elif len(image.shape) == 3:
            h, w, c = image.shape
            ret = self._dll.predict_rgb(
                self._handle, image, h, w, c, ctypes.byref(result)
            )
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")

        if ret != 0:
            raise ImageModelError(ret)

        return result.class_index, result.confidence

    # ========================================
    # 배치 처리
    # ========================================
    def predict_batch(self, images: np.ndarray) -> List[Tuple[int, float]]:
        """
        배치 이미지 분류 수행

        Args:
            images: numpy 배열 (N, H, W) - grayscale 배치

        Returns:
            [(class_index, confidence), ...] 리스트

        Raises:
            ImageModelError: DLL 에러 발생 시
        """
        if not self._handle:
            raise ImageModelError(-3)

        images = np.ascontiguousarray(images, dtype=np.float32)

        if len(images.shape) != 3:
            raise ValueError(f"Expected 3D array (N, H, W), got shape: {images.shape}")

        batch_size, height, width = images.shape
        results = (PredictionResult * batch_size)()

        ret = self._dll.predict_batch(
            self._handle, images, batch_size, height, width, results
        )

        if ret != 0:
            raise ImageModelError(ret)

        return [(r.class_index, r.confidence) for r in results]

    def predict_large_batch(self, images: np.ndarray) -> List[Tuple[int, float]]:
        """
        큰 배치를 최적 크기로 분할하여 처리

        Args:
            images: numpy 배열 (N, H, W)

        Returns:
            [(class_index, confidence), ...] 리스트
        """
        results = []
        optimal = self.optimal_batch_size

        for i in range(0, len(images), optimal):
            batch = images[i:i + optimal]
            results.extend(self.predict_batch(batch))

        return results

    # ========================================
    # 특징 추출
    # ========================================
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        이미지에서 특징 벡터 추출

        Args:
            image: numpy 배열 (any shape, flattened internally)

        Returns:
            특징 벡터 (10개 float)

        Raises:
            ImageModelError: DLL 에러 발생 시
        """
        if not self._handle:
            raise ImageModelError(-3)

        image = np.ascontiguousarray(image.flatten(), dtype=np.float32)
        features = np.zeros(10, dtype=np.float32)

        ret = self._dll.extract_features(
            self._handle, image, len(image), features, len(features)
        )

        if ret < 0:
            raise ImageModelError(ret)

        return features[:ret]


# ========================================
# 사용 예시
# ========================================
if __name__ == "__main__":
    try:
        # Context Manager 사용 (권장)
        with ImageModelDLL() as model:
            print(f"DLL Version: {model.version}")
            print(f"Optimal batch size: {model.optimal_batch_size}")
            print(f"Model valid: {model.is_valid}")

            # 단일 이미지 테스트
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            class_idx, confidence = model.predict(test_image)
            print(f"\nSingle prediction:")
            print(f"  Class: {class_idx}, Confidence: {confidence:.4f}")

            # 배치 테스트
            batch_images = np.random.randint(
                0, 255, (8, 224, 224), dtype=np.uint8
            ).astype(np.float32)
            results = model.predict_batch(batch_images)
            print(f"\nBatch prediction ({len(results)} images):")
            for i, (cls, conf) in enumerate(results[:3]):
                print(f"  Image {i}: Class {cls}, Confidence {conf:.4f}")
            print("  ...")

            # 특징 추출 테스트
            features = model.extract_features(test_image)
            print(f"\nFeatures ({len(features)} dims):")
            print(f"  {features}")

    except OSError as e:
        print(f"DLL 로드 실패: {e}")
        print("\n먼저 DLL을 빌드하세요:")
        print("  # macOS에서 Windows DLL 크로스 컴파일:")
        print("  x86_64-w64-mingw32-g++ -shared -O2 -o image_model.dll image_model.cpp -static-libgcc -static-libstdc++")
        print("\n  # 또는 Windows에서:")
        print("  mkdir build && cd build")
        print("  cmake .. -G 'Visual Studio 17 2022' -A x64")
        print("  cmake --build . --config Release")
    except ImageModelError as e:
        print(f"Model error: {e}")
