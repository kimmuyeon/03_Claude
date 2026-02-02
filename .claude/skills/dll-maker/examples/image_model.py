"""
이미지 기반 AI 모델 예시 (Python 원본)
- 입력: 이미지 데이터 (numpy array, HWC 또는 CHW 형식)
- 출력: 분류 결과 (클래스 인덱스, 신뢰도)
"""

import numpy as np
from typing import Tuple

# 간단한 특징 추출 (실제로는 CNN 등 사용)
def extract_features(image: np.ndarray) -> np.ndarray:
    """이미지에서 특징 벡터 추출"""
    # 평균, 표준편차, 히스토그램 기반 간단한 특징
    mean_val = np.mean(image)
    std_val = np.std(image)

    # 간단한 히스토그램 (8 bins)
    hist, _ = np.histogram(image.flatten(), bins=8, range=(0, 255))
    hist = hist / hist.sum()  # 정규화

    features = np.array([mean_val, std_val] + list(hist), dtype=np.float32)
    return features

# 간단한 분류기 (실제로는 학습된 가중치 사용)
def classify(features: np.ndarray, num_classes: int = 10) -> Tuple[int, float]:
    """특징 벡터로 분류 수행"""
    # 간단한 선형 변환 (실제로는 학습된 가중치)
    np.random.seed(42)  # 재현성을 위해
    weights = np.random.randn(len(features), num_classes).astype(np.float32)

    logits = features @ weights

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])

    return class_idx, confidence

# 메인 추론 함수
def predict(image: np.ndarray) -> Tuple[int, float]:
    """
    이미지 분류 수행

    Args:
        image: 이미지 데이터 (H, W) 또는 (H, W, C), uint8 또는 float

    Returns:
        (class_index, confidence) 튜플
    """
    # 전처리
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    if len(image.shape) == 3:
        # RGB to grayscale
        image = np.mean(image, axis=2)

    # 특징 추출 및 분류
    features = extract_features(image)
    class_idx, confidence = classify(features)

    return class_idx, confidence


if __name__ == "__main__":
    # 테스트
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = predict(test_image)
    print(f"Class: {result[0]}, Confidence: {result[1]:.4f}")
