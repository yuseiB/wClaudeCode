cmake_minimum_required(VERSION 3.20)
project(ising_model_2d CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Library ---
add_library(ising_lib
    src/ising_model.cpp
)
target_include_directories(ising_lib PUBLIC include)

# --- Tests ---
enable_testing()

add_executable(test_ising
    tests/test_ising.cpp
)
target_include_directories(test_ising PRIVATE include)
target_link_libraries(test_ising PRIVATE ising_lib)
add_test(NAME ising COMMAND test_ising)

# --- Example ---
add_executable(ising_sim
    examples/ising_sim.cpp
)
target_include_directories(ising_sim PRIVATE include)
target_link_libraries(ising_sim PRIVATE ising_lib)
