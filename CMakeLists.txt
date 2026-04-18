cmake_minimum_required(VERSION 3.20)

project(lp_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

option(LP_SOLVER_ENABLE_TESTS "Build tests" ON)

include(FetchContent)
FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
  URL_HASH SHA256=b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_GetProperties(Eigen3)
if(NOT Eigen3_POPULATED)
  FetchContent_Populate(Eigen3)
endif()

add_library(lp_solver STATIC
  src/util/indexed_vector.cpp
  src/util/packed_matrix.cpp
  src/linalg/eigen_factor.cpp
  src/linalg/umfpack_factor.cpp
  src/linalg/factory.cpp
  src/linalg/gilbert_peierls.cpp
  src/model/initialization.cpp
  src/presolve/presolver.cpp
  src/simplex/dual_simplex.cpp
)

target_include_directories(lp_solver
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(lp_solver SYSTEM PUBLIC ${eigen3_SOURCE_DIR})

if(LP_SOLVER_ENABLE_TESTS)
  include(CTest)
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(lp_solver_tests
    tests/test_indexed_vector.cpp
    tests/test_basis_factor_contract.cpp
    tests/test_dual_simplex.cpp
    tests/test_gilbert_peierls.cpp
    tests/test_presolve.cpp
    tests/test_big_m.cpp
  )
  target_link_libraries(lp_solver_tests PRIVATE lp_solver GTest::gtest_main)

  add_executable(lp_solver_stress
    tests/stress_factor.cpp
  )
  target_link_libraries(lp_solver_stress PRIVATE lp_solver)

  add_executable(lp_netlib_runner
    tools/netlib_runner.cpp
  )
  target_link_libraries(lp_netlib_runner PRIVATE lp_solver)

  include(GoogleTest)
  gtest_discover_tests(lp_solver_tests)
  add_test(
    NAME lp_solver_stress_smoke
    COMMAND lp_solver_stress --dim 200 --iters 200 --seed 42
  )
endif()
