set -e

cd /Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build/_deps/googletest-build/googlemock
/opt/homebrew/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
