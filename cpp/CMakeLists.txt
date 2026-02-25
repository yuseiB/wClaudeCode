cmake_minimum_required(VERSION 3.20)
project(mathphys CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Library ---
add_library(mathphys_lib
    src/numerics.cpp
)
target_include_directories(mathphys_lib PUBLIC include)

# --- Tests ---
enable_testing()

add_executable(test_numerics
    tests/test_numerics.cpp
)
target_include_directories(test_numerics PRIVATE include)
target_link_libraries(test_numerics PRIVATE mathphys_lib)

add_test(NAME numerics COMMAND test_numerics)
