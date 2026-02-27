cmake_minimum_required(VERSION 3.16)
project(em_cavity_waveguide CXX)

# C++20 を使用 (std::cyl_bessel_j, constexpr など)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 最適化フラグ
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# インクルードディレクトリ
include_directories(include)

# ── 共有ライブラリ ────────────────────────────────────────────────────────────
add_library(em_lib STATIC
    src/em_cavity.cpp
    src/em_waveguide.cpp
)
target_include_directories(em_lib PUBLIC include)

# ── シミュレーション実行ファイル ──────────────────────────────────────────────
add_executable(em_sim examples/em_sim.cpp)
target_link_libraries(em_sim PRIVATE em_lib)

# ── テスト実行ファイル ────────────────────────────────────────────────────────
enable_testing()

add_executable(test_em tests/test_em.cpp)
target_link_libraries(test_em PRIVATE em_lib)

# CTest に登録
add_test(NAME EMCavityWaveguideTests COMMAND test_em)
