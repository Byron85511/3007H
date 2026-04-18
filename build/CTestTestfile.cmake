# CMake generated Testfile for 
# Source directory: /Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project
# Build directory: /Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build/lp_solver_tests[1]_include.cmake")
add_test([=[lp_solver_stress_smoke]=] "/Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build/lp_solver_stress" "--dim" "200" "--iters" "200" "--seed" "42")
set_tests_properties([=[lp_solver_stress_smoke]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/CMakeLists.txt;79;add_test;/Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
