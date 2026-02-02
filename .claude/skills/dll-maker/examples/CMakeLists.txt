cmake_minimum_required(VERSION 3.10)
project(ImageModel VERSION 1.0.0 LANGUAGES CXX)

# C++ 17 표준
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# DLL 빌드
add_library(image_model SHARED
    image_model.cpp
)

# 헤더 파일 경로 (현재 디렉토리)
target_include_directories(image_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Release 빌드 최적화
if(MSVC)
    target_compile_options(image_model PRIVATE
        $<$<CONFIG:Release>:/O2 /GL /fp:fast>
    )
    target_link_options(image_model PRIVATE
        $<$<CONFIG:Release>:/LTCG>
    )
else()
    target_compile_options(image_model PRIVATE
        $<$<CONFIG:Release>:-O3 -ffast-math -flto>
    )
endif()

# DLL을 Python 래퍼와 같은 위치에 출력
set_target_properties(image_model PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
)
