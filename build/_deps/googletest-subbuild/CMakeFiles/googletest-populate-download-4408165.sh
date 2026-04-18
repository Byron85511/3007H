set -e

cd /Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build/_deps
/opt/homebrew/bin/cmake -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE -P /Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/download-googletest-populate.cmake
/opt/homebrew/bin/cmake -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE -P /Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/verify-googletest-populate.cmake
/opt/homebrew/bin/cmake -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE -P /Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/extract-googletest-populate.cmake
/opt/homebrew/bin/cmake -E touch /Users/lilliansun/CUHKSZ/Y2/T2/MAT3007H/project/mat3007h_project/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/googletest-populate-download
