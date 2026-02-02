cmake_minimum_required(VERSION 3.10)
project(MyDLL VERSION 1.0.0 LANGUAGES CXX)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows에서 모든 심볼 내보내기 (extern "C" 사용 시 불필요하지만 안전을 위해)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)

# DLL 빌드
add_library(mydll SHARED
    src/mydll.cpp
)

# 헤더 파일 경로
target_include_directories(mydll
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Release 빌드 최적화
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(mydll PRIVATE /O2 /GL)
        target_link_options(mydll PRIVATE /LTCG)
    else()
        target_compile_options(mydll PRIVATE -O3 -flto)
        target_link_options(mydll PRIVATE -flto)
    endif()
endif()

# 출력 디렉토리 설정
set_target_properties(mydll PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# 설치 규칙
install(TARGETS mydll
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)
