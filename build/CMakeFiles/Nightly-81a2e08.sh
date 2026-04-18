set -e

cd /Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build
/opt/homebrew/bin/ctest -DMODEL=Nightly -S CMakeFiles/CTestScript.cmake -V
