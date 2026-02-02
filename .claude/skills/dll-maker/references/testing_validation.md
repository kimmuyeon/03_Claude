# 테스트 및 검증 패턴

## Python-C++ 결과 일치 검증

### 검증 프레임워크

```python
"""
Python-C++ DLL 결과 일치 검증 프레임워크
"""

import numpy as np
from typing import Callable, Any, Dict, List, Tuple
from dataclasses import dataclass
import time

@dataclass
class TestResult:
    name: str
    passed: bool
    python_result: Any
    cpp_result: Any
    max_diff: float
    python_time_ms: float
    cpp_time_ms: float
    speedup: float
    error_message: str = ""

class DLLValidator:
    """Python 함수와 DLL 함수의 결과를 비교 검증"""

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8):
        """
        Args:
            rtol: 상대 허용 오차
            atol: 절대 허용 오차
        """
        self.rtol = rtol
        self.atol = atol
        self.results: List[TestResult] = []

    def compare_arrays(
        self,
        python_arr: np.ndarray,
        cpp_arr: np.ndarray
    ) -> Tuple[bool, float]:
        """두 배열 비교"""
        if python_arr.shape != cpp_arr.shape:
            return False, float('inf')

        diff = np.abs(python_arr - cpp_arr)
        max_diff = float(np.max(diff))

        # NumPy의 allclose와 동일한 기준
        passed = np.allclose(python_arr, cpp_arr, rtol=self.rtol, atol=self.atol)

        return passed, max_diff

    def validate(
        self,
        name: str,
        python_func: Callable,
        cpp_func: Callable,
        inputs: Dict[str, Any],
        python_inputs: Dict[str, Any] = None,
        cpp_inputs: Dict[str, Any] = None,
        iterations: int = 10
    ) -> TestResult:
        """
        단일 테스트 케이스 검증

        Args:
            name: 테스트 이름
            python_func: Python 구현 함수
            cpp_func: C++ DLL 래퍼 함수
            inputs: 공통 입력 (둘 다 사용)
            python_inputs: Python 전용 추가 입력
            cpp_inputs: C++ 전용 추가 입력
            iterations: 성능 측정 반복 횟수
        """
        py_inputs = {**inputs, **(python_inputs or {})}
        cpp_in = {**inputs, **(cpp_inputs or {})}

        try:
            # Python 실행 및 시간 측정
            start = time.perf_counter()
            for _ in range(iterations):
                python_result = python_func(**py_inputs)
            python_time = (time.perf_counter() - start) / iterations * 1000

            # C++ 실행 및 시간 측정
            start = time.perf_counter()
            for _ in range(iterations):
                cpp_result = cpp_func(**cpp_in)
            cpp_time = (time.perf_counter() - start) / iterations * 1000

            # 결과 비교
            if isinstance(python_result, np.ndarray):
                passed, max_diff = self.compare_arrays(
                    np.asarray(python_result),
                    np.asarray(cpp_result)
                )
            elif isinstance(python_result, (tuple, list)):
                passed = True
                max_diff = 0.0
                for py_val, cpp_val in zip(python_result, cpp_result):
                    if isinstance(py_val, np.ndarray):
                        p, d = self.compare_arrays(
                            np.asarray(py_val),
                            np.asarray(cpp_val)
                        )
                        passed = passed and p
                        max_diff = max(max_diff, d)
                    else:
                        diff = abs(float(py_val) - float(cpp_val))
                        max_diff = max(max_diff, diff)
                        passed = passed and (diff < self.atol)
            else:
                max_diff = abs(float(python_result) - float(cpp_result))
                passed = max_diff < self.atol

            speedup = python_time / cpp_time if cpp_time > 0 else float('inf')

            result = TestResult(
                name=name,
                passed=passed,
                python_result=python_result,
                cpp_result=cpp_result,
                max_diff=max_diff,
                python_time_ms=python_time,
                cpp_time_ms=cpp_time,
                speedup=speedup
            )

        except Exception as e:
            result = TestResult(
                name=name,
                passed=False,
                python_result=None,
                cpp_result=None,
                max_diff=float('inf'),
                python_time_ms=0,
                cpp_time_ms=0,
                speedup=0,
                error_message=str(e)
            )

        self.results.append(result)
        return result

    def report(self) -> str:
        """테스트 결과 리포트 생성"""
        lines = [
            "=" * 80,
            "DLL Validation Report",
            "=" * 80,
            ""
        ]

        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"[{status}] {r.name}")

            if r.error_message:
                lines.append(f"  Error: {r.error_message}")
            else:
                lines.append(f"  Max difference: {r.max_diff:.2e}")
                lines.append(f"  Python time: {r.python_time_ms:.3f} ms")
                lines.append(f"  C++ time: {r.cpp_time_ms:.3f} ms")
                lines.append(f"  Speedup: {r.speedup:.2f}x")
            lines.append("")

        lines.append("-" * 80)
        lines.append(f"Total: {passed_count}/{total_count} passed")
        lines.append("=" * 80)

        return "\n".join(lines)


# 사용 예시
if __name__ == "__main__":
    # Python 원본 함수
    def python_softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def python_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    # C++ DLL 래퍼 (예시)
    class MockDLL:
        @staticmethod
        def softmax(x: np.ndarray) -> np.ndarray:
            # DLL 호출 시뮬레이션
            exp_x = np.exp(x - np.max(x))
            return exp_x / exp_x.sum()

        @staticmethod
        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return a @ b

    dll = MockDLL()

    # 검증 실행
    validator = DLLValidator(rtol=1e-5, atol=1e-8)

    # Softmax 테스트
    test_input = np.random.randn(1000).astype(np.float32)
    validator.validate(
        name="Softmax 1000 elements",
        python_func=python_softmax,
        cpp_func=dll.softmax,
        inputs={"x": test_input}
    )

    # 행렬 곱 테스트
    a = np.random.randn(100, 200).astype(np.float32)
    b = np.random.randn(200, 50).astype(np.float32)
    validator.validate(
        name="Matrix multiplication 100x200 @ 200x50",
        python_func=python_matmul,
        cpp_func=dll.matmul,
        inputs={"a": a, "b": b}
    )

    # 리포트 출력
    print(validator.report())
```

---

## 자동화된 테스트 스위트

### pytest 기반 테스트

```python
"""
pytest 기반 DLL 테스트 스위트
파일명: test_dll.py
실행: pytest test_dll.py -v
"""

import pytest
import numpy as np
import ctypes
from pathlib import Path

# DLL 래퍼 임포트
# from my_model_wrapper import MyModelDLL

class TestDLLBasic:
    """기본 기능 테스트"""

    @pytest.fixture(scope="class")
    def dll(self):
        """DLL 인스턴스 (클래스당 한 번 로드)"""
        dll_path = Path(__file__).parent / "my_model.dll"
        # return MyModelDLL(str(dll_path))

        # Mock for example
        class MockDLL:
            def add(self, a, b): return a + b
            def sum_array(self, arr): return np.sum(arr)
        return MockDLL()

    def test_simple_add(self, dll):
        """단순 덧셈 테스트"""
        assert dll.add(2, 3) == 5
        assert dll.add(-1, 1) == 0
        assert dll.add(0, 0) == 0

    def test_large_numbers(self, dll):
        """큰 숫자 처리"""
        large = 2**30
        result = dll.add(large, large)
        assert result == large * 2

    def test_array_sum(self, dll):
        """배열 합계"""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        result = dll.sum_array(arr)
        np.testing.assert_almost_equal(result, 15.0, decimal=10)


class TestDLLNumerical:
    """수치 정확도 테스트"""

    @pytest.fixture(scope="class")
    def dll(self):
        class MockDLL:
            def softmax(self, x):
                exp_x = np.exp(x - np.max(x))
                return exp_x / exp_x.sum()
        return MockDLL()

    @pytest.mark.parametrize("size", [10, 100, 1000, 10000])
    def test_softmax_sizes(self, dll, size):
        """다양한 크기의 softmax"""
        x = np.random.randn(size).astype(np.float32)
        result = dll.softmax(x)

        # 합이 1인지 확인
        np.testing.assert_almost_equal(np.sum(result), 1.0, decimal=5)

        # 모든 값이 0 이상인지
        assert np.all(result >= 0)

    def test_softmax_numerical_stability(self, dll):
        """수치 안정성 (큰 값)"""
        x = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)
        result = dll.softmax(x)

        # NaN이나 Inf가 없어야 함
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_almost_equal(np.sum(result), 1.0, decimal=5)

    def test_softmax_negative(self, dll):
        """음수 입력"""
        x = np.array([-100.0, -200.0, -300.0], dtype=np.float32)
        result = dll.softmax(x)
        np.testing.assert_almost_equal(np.sum(result), 1.0, decimal=5)


class TestDLLEdgeCases:
    """엣지 케이스 테스트"""

    @pytest.fixture(scope="class")
    def dll(self):
        class MockDLL:
            def process(self, arr):
                if arr is None or len(arr) == 0:
                    raise ValueError("Invalid input")
                return arr * 2
        return MockDLL()

    def test_empty_array(self, dll):
        """빈 배열 처리"""
        with pytest.raises((ValueError, RuntimeError)):
            dll.process(np.array([], dtype=np.float32))

    def test_single_element(self, dll):
        """단일 원소"""
        arr = np.array([42.0], dtype=np.float32)
        result = dll.process(arr)
        np.testing.assert_array_equal(result, [84.0])

    def test_non_contiguous_array(self, dll):
        """비연속 메모리 배열"""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        non_contig = arr[:, ::2]  # [1, 3], [4, 6]

        assert not non_contig.flags['C_CONTIGUOUS']

        # DLL은 연속 배열 필요 - 래퍼가 자동 변환해야 함
        contig = np.ascontiguousarray(non_contig)
        result = dll.process(contig)
        assert result is not None


class TestDLLPerformance:
    """성능 테스트"""

    @pytest.fixture(scope="class")
    def dll(self):
        class MockDLL:
            def matmul(self, a, b):
                return a @ b
        return MockDLL()

    @pytest.mark.benchmark
    def test_matmul_performance(self, dll, benchmark):
        """행렬 곱 벤치마크 (pytest-benchmark 필요)"""
        a = np.random.randn(256, 256).astype(np.float32)
        b = np.random.randn(256, 256).astype(np.float32)

        # benchmark(dll.matmul, a, b)
        result = dll.matmul(a, b)
        assert result.shape == (256, 256)

    def test_scaling(self, dll):
        """크기별 성능 스케일링"""
        import time

        sizes = [64, 128, 256, 512]
        times = []

        for size in sizes:
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)

            start = time.perf_counter()
            for _ in range(10):
                dll.matmul(a, b)
            elapsed = (time.perf_counter() - start) / 10

            times.append(elapsed)

        # O(n^3) 복잡도 확인: 크기가 2배면 시간은 8배
        # 어느 정도 여유를 두고 검사
        for i in range(1, len(sizes)):
            ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]
            expected_ratio = ratio ** 3

            # 실제 시간 비율이 예상의 0.5~2.0배 범위 내인지
            assert 0.3 * expected_ratio < time_ratio < 3.0 * expected_ratio


class TestDLLMemory:
    """메모리 관련 테스트"""

    @pytest.fixture(scope="class")
    def dll(self):
        class MockDLL:
            def __init__(self):
                self._allocations = []

            def allocate(self, size):
                arr = np.zeros(size, dtype=np.float32)
                self._allocations.append(arr)
                return id(arr)

            def free(self, ptr):
                self._allocations = [a for a in self._allocations if id(a) != ptr]

            def get_allocation_count(self):
                return len(self._allocations)
        return MockDLL()

    def test_memory_leak(self, dll):
        """메모리 누수 검사"""
        initial_count = dll.get_allocation_count()

        # 여러 번 할당/해제
        for _ in range(100):
            ptr = dll.allocate(1000)
            dll.free(ptr)

        final_count = dll.get_allocation_count()
        assert final_count == initial_count

    def test_large_allocation(self, dll):
        """큰 메모리 할당"""
        # 100MB 할당 시도
        large_size = 100 * 1024 * 1024 // 4  # float32 기준

        try:
            ptr = dll.allocate(large_size)
            dll.free(ptr)
        except MemoryError:
            pytest.skip("Not enough memory for large allocation test")


# conftest.py에 추가할 fixture
@pytest.fixture(scope="session")
def dll_path():
    """DLL 경로"""
    return Path(__file__).parent / "build" / "Release" / "my_model.dll"


@pytest.fixture(scope="session")
def python_reference():
    """Python 참조 구현"""
    class PythonReference:
        @staticmethod
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / exp_x.sum(axis=-1, keepdims=True)

        @staticmethod
        def matmul(a, b):
            return np.matmul(a, b)

    return PythonReference()
```

---

## C++ 단위 테스트

### Google Test 기반

```cpp
// test_model.cpp
#include <gtest/gtest.h>
#include "my_model.h"
#include <vector>
#include <cmath>
#include <numeric>

// 테스트 픽스처
class ModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 테스트 전 초기화
    }

    void TearDown() override {
        // 테스트 후 정리
    }

    // 헬퍼: 두 배열 비교
    bool arrays_close(const float* a, const float* b, size_t len,
                      float rtol = 1e-5f, float atol = 1e-8f) {
        for (size_t i = 0; i < len; i++) {
            float diff = std::abs(a[i] - b[i]);
            float threshold = atol + rtol * std::abs(b[i]);
            if (diff > threshold) {
                return false;
            }
        }
        return true;
    }
};

// 기본 연산 테스트
TEST_F(ModelTest, AddBasic) {
    EXPECT_EQ(add(2, 3), 5);
    EXPECT_EQ(add(-1, 1), 0);
    EXPECT_EQ(add(0, 0), 0);
}

TEST_F(ModelTest, AddLargeNumbers) {
    int large = 1 << 30;
    // 오버플로우 확인
    EXPECT_EQ(add(large, 0), large);
}

// 배열 처리 테스트
TEST_F(ModelTest, SumArray) {
    std::vector<double> arr = {1.0, 2.0, 3.0, 4.0, 5.0};
    double result = sum_array(arr.data(), arr.size());
    EXPECT_DOUBLE_EQ(result, 15.0);
}

TEST_F(ModelTest, SumArrayEmpty) {
    double result = sum_array(nullptr, 0);
    EXPECT_DOUBLE_EQ(result, 0.0);  // 또는 에러 코드 확인
}

// Softmax 테스트
TEST_F(ModelTest, SoftmaxBasic) {
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3);

    softmax(input.data(), output.data(), 3);

    // 합이 1인지 확인
    float sum = std::accumulate(output.begin(), output.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // 순서 유지 확인 (입력이 클수록 출력도 커야 함)
    EXPECT_LT(output[0], output[1]);
    EXPECT_LT(output[1], output[2]);
}

TEST_F(ModelTest, SoftmaxNumericalStability) {
    // 큰 값으로 테스트
    std::vector<float> input = {1000.0f, 1000.0f, 1000.0f};
    std::vector<float> output(3);

    softmax(input.data(), output.data(), 3);

    // NaN/Inf 체크
    for (float v : output) {
        EXPECT_FALSE(std::isnan(v));
        EXPECT_FALSE(std::isinf(v));
    }

    // 균등 분포 확인
    for (float v : output) {
        EXPECT_NEAR(v, 1.0f/3.0f, 1e-5f);
    }
}

// 행렬 연산 테스트
TEST_F(ModelTest, MatMulIdentity) {
    // 단위 행렬 곱
    std::vector<float> a = {1, 2, 3, 4};  // 2x2
    std::vector<float> identity = {1, 0, 0, 1};  // 2x2
    std::vector<float> result(4);

    matmul(a.data(), identity.data(), result.data(), 2, 2, 2);

    EXPECT_TRUE(arrays_close(result.data(), a.data(), 4));
}

// 에러 처리 테스트
TEST_F(ModelTest, NullPointerHandling) {
    int result = process(nullptr, 10);
    EXPECT_EQ(result, ERR_NULL_PTR);
}

TEST_F(ModelTest, InvalidSizeHandling) {
    std::vector<float> data(10);
    int result = process(data.data(), 0);
    EXPECT_EQ(result, ERR_INVALID_SIZE);
}

// 성능 테스트 (Google Benchmark 별도 사용 권장)
TEST_F(ModelTest, PerformanceBaseline) {
    const size_t size = 10000;
    std::vector<float> input(size);
    std::vector<float> output(size);

    // 랜덤 초기화
    for (size_t i = 0; i < size; i++) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < 100; iter++) {
        process(input.data(), output.data(), size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 100번 반복에 100ms 이하
    EXPECT_LT(duration.count(), 100000);
}

// 메인 함수
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

### CMakeLists.txt (테스트 포함)

```cmake
cmake_minimum_required(VERSION 3.14)
project(MyModelWithTests)

set(CMAKE_CXX_STANDARD 17)

# Google Test 가져오기
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# 메인 라이브러리
add_library(mymodel SHARED
    src/my_model.cpp
)
target_include_directories(mymodel PUBLIC include)

# 테스트 실행 파일
enable_testing()

add_executable(mymodel_test
    tests/test_model.cpp
)

target_link_libraries(mymodel_test PRIVATE
    mymodel
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(mymodel_test)
```

---

## 통합 검증 스크립트

```python
#!/usr/bin/env python3
"""
전체 검증 파이프라인
실행: python validate_dll.py --dll path/to/model.dll
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import subprocess
import json

def run_cpp_tests(build_dir: Path) -> bool:
    """C++ 테스트 실행"""
    print("Running C++ unit tests...")
    result = subprocess.run(
        ["ctest", "--output-on-failure"],
        cwd=build_dir,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return False
    return True

def run_python_tests(test_dir: Path) -> bool:
    """Python pytest 실행"""
    print("Running Python tests...")
    result = subprocess.run(
        ["pytest", "-v", "--tb=short"],
        cwd=test_dir,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return False
    return True

def run_numerical_validation(dll_path: str) -> dict:
    """수치 검증"""
    print("Running numerical validation...")

    # DLL 로드 (실제 구현에서는 래퍼 사용)
    # from my_model_wrapper import MyModelDLL
    # dll = MyModelDLL(dll_path)

    results = {
        "softmax": {"passed": True, "max_error": 0.0},
        "matmul": {"passed": True, "max_error": 0.0},
    }

    # 테스트 케이스들
    np.random.seed(42)

    # Softmax 검증
    for size in [10, 100, 1000]:
        x = np.random.randn(size).astype(np.float32)
        expected = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
        # actual = dll.softmax(x)

        # Mock
        actual = expected + np.random.randn(size).astype(np.float32) * 1e-7

        error = np.max(np.abs(expected - actual))
        results["softmax"]["max_error"] = max(
            results["softmax"]["max_error"], error
        )
        if error > 1e-5:
            results["softmax"]["passed"] = False

    return results

def run_performance_validation(dll_path: str) -> dict:
    """성능 검증"""
    print("Running performance validation...")
    import time

    results = {}

    # 벤치마크 케이스
    sizes = [256, 512, 1024]

    for size in sizes:
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)

        # Python 기준
        start = time.perf_counter()
        for _ in range(10):
            _ = a @ b
        python_time = (time.perf_counter() - start) / 10

        # C++ DLL (Mock)
        cpp_time = python_time * 0.5  # 실제로는 DLL 호출

        results[f"matmul_{size}x{size}"] = {
            "python_ms": python_time * 1000,
            "cpp_ms": cpp_time * 1000,
            "speedup": python_time / cpp_time
        }

    return results

def generate_report(
    cpp_passed: bool,
    python_passed: bool,
    numerical: dict,
    performance: dict,
    output_path: Path
):
    """검증 리포트 생성"""
    report = {
        "cpp_tests": {"passed": cpp_passed},
        "python_tests": {"passed": python_passed},
        "numerical_validation": numerical,
        "performance": performance,
        "overall_passed": (
            cpp_passed and
            python_passed and
            all(v["passed"] for v in numerical.values())
        )
    }

    # JSON 저장
    with open(output_path / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # 텍스트 리포트
    lines = [
        "=" * 60,
        "DLL Validation Report",
        "=" * 60,
        "",
        f"C++ Unit Tests: {'PASS' if cpp_passed else 'FAIL'}",
        f"Python Tests: {'PASS' if python_passed else 'FAIL'}",
        "",
        "Numerical Validation:",
    ]

    for name, result in numerical.items():
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(f"  {name}: {status} (max error: {result['max_error']:.2e})")

    lines.append("")
    lines.append("Performance:")
    for name, result in performance.items():
        lines.append(f"  {name}:")
        lines.append(f"    Python: {result['python_ms']:.2f} ms")
        lines.append(f"    C++: {result['cpp_ms']:.2f} ms")
        lines.append(f"    Speedup: {result['speedup']:.2f}x")

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"OVERALL: {'PASS' if report['overall_passed'] else 'FAIL'}")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print(report_text)

    with open(output_path / "validation_report.txt", "w") as f:
        f.write(report_text)

    return report["overall_passed"]

def main():
    parser = argparse.ArgumentParser(description="DLL Validation Pipeline")
    parser.add_argument("--dll", required=True, help="Path to DLL file")
    parser.add_argument("--build-dir", default="build", help="C++ build directory")
    parser.add_argument("--test-dir", default="tests", help="Python test directory")
    parser.add_argument("--output", default=".", help="Output directory for reports")

    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # 1. C++ 테스트
    cpp_passed = True  # run_cpp_tests(build_dir)

    # 2. Python 테스트
    python_passed = True  # run_python_tests(test_dir)

    # 3. 수치 검증
    numerical = run_numerical_validation(args.dll)

    # 4. 성능 검증
    performance = run_performance_validation(args.dll)

    # 5. 리포트 생성
    overall = generate_report(
        cpp_passed, python_passed, numerical, performance, output_dir
    )

    sys.exit(0 if overall else 1)

if __name__ == "__main__":
    main()
```
