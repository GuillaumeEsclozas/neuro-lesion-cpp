cmake_minimum_required(VERSION 3.18)
project(brain_lesion_seg LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# ONNX Runtime prebuilt binaries
include(FetchContent)
set(ORT_VERSION "1.17.1")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-x64-${ORT_VERSION}.zip")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-osx-x86_64-${ORT_VERSION}.tgz")
else()
    message(FATAL_ERROR "Unsupported platform. Set ONNXRUNTIME_ROOT manually.")
endif()

FetchContent_Declare(onnxruntime URL ${ORT_URL})
FetchContent_MakeAvailable(onnxruntime)

set(ORT_ROOT "${onnxruntime_SOURCE_DIR}")
set(ORT_INCLUDE "${ORT_ROOT}/include")

if(WIN32)
    set(ORT_LIB "${ORT_ROOT}/lib/onnxruntime.lib")
else()
    file(GLOB ORT_LIB "${ORT_ROOT}/lib/libonnxruntime.so*" "${ORT_ROOT}/lib/libonnxruntime.dylib*")
    list(GET ORT_LIB 0 ORT_LIB)
endif()

find_package(ZLIB REQUIRED)

# Library
set(LIB_SOURCES
    src/NiftiIO.cpp
    src/Preprocessor.cpp
    src/InferenceEngine.cpp
    src/Postprocessor.cpp
    include/znzlib.c
)

add_library(lesion_seg_lib STATIC ${LIB_SOURCES})
target_include_directories(lesion_seg_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${ORT_INCLUDE}
)
target_link_libraries(lesion_seg_lib PUBLIC ${ORT_LIB} ZLIB::ZLIB)

# Main binary
add_executable(brain_lesion_seg src/main.cpp)
target_link_libraries(brain_lesion_seg PRIVATE lesion_seg_lib)

if(WIN32)
    file(GLOB ORT_DLLS "${ORT_ROOT}/lib/*.dll")
    foreach(dll ${ORT_DLLS})
        add_custom_command(TARGET brain_lesion_seg POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:brain_lesion_seg>"
        )
    endforeach()
endif()

# Tests
enable_testing()

add_executable(test_preprocessor tests/test_preprocessor.cpp)
target_link_libraries(test_preprocessor PRIVATE lesion_seg_lib)
add_test(NAME preprocessor_tests COMMAND test_preprocessor)

add_executable(test_postprocessor tests/test_postprocessor.cpp)
target_link_libraries(test_postprocessor PRIVATE lesion_seg_lib)
add_test(NAME postprocessor_tests COMMAND test_postprocessor)
