#!/bin/sh
# make-so.sh <path to amd_fidelityfx_framegeneration_vk.dll> [output dir]
#
# Builds amd_fidelityfx_framegeneration_vk.so from YOUR OWN copy of AMD's DLL (FSR SDK 2.4.0, the Vulkan frame
# generation provider). The DLL is read as data only (never run): its shaders and model weights are extracted at build
# time and embedded in the .so. That is why this repository does not ship a binary: build your own, for your own use.
# Get the DLL from AMD's FidelityFX SDK 2.4.0 (https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK), or from
# a game that ships it.
#
#   ./make-so.sh /path/to/amd_fidelityfx_framegeneration_vk.dll        -> ./dist/amd_fidelityfx_framegeneration_vk.so
#   ./make-so.sh /path/to/amd_fidelityfx_framegeneration_vk.dll /opt/x -> /opt/x/amd_fidelityfx_framegeneration_vk.so
#
# Needs: cmake, a C++17 compiler (clang++ or g++), python3, and the Vulkan headers (vulkan/vulkan_core.h).
set -eu

NAME=amd_fidelityfx_framegeneration_vk
HERE=$(cd "$(dirname "$0")" && pwd)

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
DLL=$1
OUT=${2:-$HERE/dist}
BUILD=$HERE/build-so

die() { echo "make-so.sh: $*" >&2; exit 1; }

[ -f "$DLL" ] || die "not a file: $DLL"
DLL=$(realpath "$DLL")

for tool in cmake python3; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool is required but not installed"
done
if [ -z "${CXX:-}" ]; then
    if command -v clang++ >/dev/null 2>&1; then CXX=clang++
    elif command -v g++ >/dev/null 2>&1; then CXX=g++
    else die "a C++ compiler (clang++ or g++) is required"; fi
fi
export CXX
printf '#include <vulkan/vulkan_core.h>\n' | $CXX -x c++ -fsyntax-only - 2>/dev/null \
    || die "the Vulkan headers were not found (package: vulkan-headers / libvulkan-dev)"

# Check the DLL is the exact build this port was made for, before spending time on a build.
WANT=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['dll_sha256'])" "$HERE/data/model.json")
HAVE=$(sha256sum "$DLL" | cut -d' ' -f1)
if [ "$HAVE" != "$WANT" ]; then
    die "wrong DLL.
  expected SHA-256 $WANT  (FSR SDK 2.4.0, FSR Frame Generation 4.0.2)
  got      SHA-256 $HAVE
Other SDK versions have different shaders and weights and are not supported."
fi
echo "DLL ok: $DLL"

cmake -S "$HERE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DFSR4FG_AMD_DLL="$DLL" -DFSR4FG_BUILD_HARNESS=OFF -DFSR4FG_BUILD_HARNESS_EXE=OFF >/dev/null
cmake --build "$BUILD" --target fsr4fg -j"$(nproc)"

mkdir -p "$OUT"
cp "$BUILD/$NAME.so" "$OUT/$NAME.so"
echo
echo "Built: $OUT/$NAME.so"
echo "This file embeds AMD's shaders and weights. Keep it for your own use; do not redistribute it."
