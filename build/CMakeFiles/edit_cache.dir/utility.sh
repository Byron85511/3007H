set -e

cd /Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build
/opt/homebrew/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
