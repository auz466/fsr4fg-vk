#!/bin/sh
# build.sh [path to AMD's amd_fidelityfx_framegeneration_vk.dll]: configures and builds with CMake into build/.
# The DLL (FSR SDK 2.4.0) is needed the first time, or to switch to another copy; it is remembered after that.
#   build/amd_fidelityfx_framegeneration_vk.so, build/trace, and build/wine/trace.exe (with mingw-w64)
set -e
cd "$(dirname "$0")"
if [ -n "$1" ]; then set -- "-DFSR4FG_AMD_DLL=$(realpath "$1")"; fi
# (a fresh build directory gets clang, as before, unless CXX says otherwise)
if [ ! -f build/CMakeCache.txt ] && [ -z "$CXX" ] && command -v clang++ >/dev/null; then export CXX=clang++; fi
cmake -S . -B build "$@"
cmake --build build -j"$(nproc)"
ls -la build/amd_fidelityfx_framegeneration_vk.so build/trace
