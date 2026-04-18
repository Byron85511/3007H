set -e

cd /Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build
/opt/homebrew/bin/cmake -D TEST_TARGET=lp_solver_tests -D TEST_EXECUTABLE=/Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build/lp_solver_tests -D TEST_EXECUTOR= -D TEST_WORKING_DIR=/Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build -D TEST_EXTRA_ARGS= -D TEST_PROPERTIES= -D TEST_PREFIX= -D TEST_SUFFIX= -D TEST_FILTER= -D NO_PRETTY_TYPES=FALSE -D NO_PRETTY_VALUES=FALSE -D TEST_LIST=lp_solver_tests_TESTS -D CTEST_FILE=/Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build/lp_solver_tests[1]_tests.cmake -D TEST_DISCOVERY_TIMEOUT=5 -D TEST_DISCOVERY_EXTRA_ARGS= -D TEST_XML_OUTPUT_DIR= -P /opt/homebrew/share/cmake/Modules/GoogleTestAddTests.cmake
