#!/bin/sh
# validation.sh [W H RW RH frames]: the port under VK_LAYER_KHRONOS_validation, with synchronisation validation, on
# the moving camera, twice. Needs the Khronos validation layer (vulkan-validation-layers).
#   1. FSR4FG_ALL_BARRIERS=1 (every pass keeps its barrier): fails if the layer reports anything but the known below.
#   2. as shipped: the border passes that share the barrier of the pass before them (border_joins in fsr4fg.cpp) write
#      the scratch buffer with no barrier after it, which the layer cannot see is safe (it knows the buffer, not the
#      words; tools/probe_borders.py measured them). Only those may be reported: write-after-write on the scratch
#      buffer (binding 128), at most one per shared barrier a frame. Anything else fails.
#   FSR4FG_BUILD   the build directory (default: <repo>/build)
#   FSR4FG_RE      where the run goes (default: <repo>/re), as validation/
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
build=${FSR4FG_BUILD:-$root/build}; out=${FSR4FG_RE:-$root/re}/validation
run() {   # run <dir>: the harness under the layer, into <dir>
rm -rf "$1"; mkdir -p "$1"
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_KHRONOS_VALIDATION_VALIDATE_CORE=true
export VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true
export VK_KHRONOS_VALIDATION_SYNCVAL_FULL_VALIDATION=true
export VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC=true
export VK_KHRONOS_VALIDATION_VALIDATE_BEST_PRACTICES=false
export VK_KHRONOS_VALIDATION_REPORT_FLAGS=error,warn
export VK_KHRONOS_VALIDATION_DEBUG_ACTION=VK_DBG_LAYER_ACTION_LOG_MSG
export VK_KHRONOS_VALIDATION_LOG_FILENAME="$1/validation.txt"
export VK_KHRONOS_VALIDATION_DUPLICATE_MESSAGE_LIMIT=${limit:-3}
TRACE_CAMERA=${TRACE_CAMERA-0.5} TRACE_RESET_AT=${TRACE_RESET_AT-2} TRACE_DUMP_ALL= FSR4FG_LOG="$1/fsr4fg.log" \
    "$build/trace" "$build/amd_fidelityfx_framegeneration_vk.so" "$1" "$W" "$H" "$RW" "$RH" "$FRAMES" 99 \
    > "$1/stdout.txt" 2>&1 || { echo "the harness failed: $1/stdout.txt"; exit 1; }
}
W=${1:-1920} H=${2:-1080} RW=${3:-1286} RH=${4:-724} FRAMES=${5:-4}
FSR4FG_ALL_BARRIERS=1 run "$out"
# (with VK_INSTANCE_LAYERS naming a layer that is missing, instance creation fails: a clean run did load it)
#
# Known, and AMD's: 6 of AMD's network shaders declare SPV_NV_cooperative_matrix2 and CooperativeMatrixConversionsNV
# (OpFConvert from one cooperative-matrix use to another), which RADV compiles but does not advertise. They are
# reported at vkCreateShaderModule as 08737, 08740 and 08742. Fixing them would mean rewriting AMD's network.
# Every other message fails the run.
known='VUID-VkShaderModuleCreateInfo-pCode-087(37|40|42)'
other=$(grep -E '^Validation (Error|Warning)' "$out/validation.txt" 2>/dev/null | grep -vE "$known" || true)
if [ -n "$other" ]; then
    echo "validation reported problems: $out/validation.txt"
    echo "$other" | grep -oE 'VUID-[A-Za-z0-9_-]+|SYNC-[A-Z0-9-]+' | sort | uniq -c | sort -rn | head -20
    exit 1
fi
if grep -E "$known" "$out/validation.txt" >/dev/null 2>&1 && ! grep -E "cooperative_matrix2|CooperativeMatrixConversionsNV|Expected Use of Matrix type" "$out/validation.txt" >/dev/null; then
    echo "a known VUID with an unexpected cause: $out/validation.txt"; exit 1
fi
echo "validation clean with every barrier ($(grep -c 'generate -> 0' "$out/stdout.txt") frames; AMD's cooperative-matrix shaders reported as known)"

# 2. as shipped (every message kept, to count them)
joined="$out-joined"
FSR4FG_ALL_BARRIERS=0 limit=0 run "$joined"
n=$(sed -n 's/^barriers: \([0-9]*\) of .*/\1/p' "$joined/fsr4fg.log" | head -1)
frames=$(grep -c 'generate -> 0' "$joined/stdout.txt")
waw=$(grep -c 'SYNC-HAZARD-WRITE-AFTER-WRITE' "$joined/validation.txt" 2>/dev/null || true)
other=$(grep -E '^Validation (Error|Warning)' "$joined/validation.txt" 2>/dev/null | grep -vE "$known|SYNC-HAZARD-WRITE-AFTER-WRITE" || true)
on128=$(grep -A2 'SYNC-HAZARD-WRITE-AFTER-WRITE' "$joined/validation.txt" 2>/dev/null | grep -c 'descriptor binding 128 (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)' || true)
if [ -n "$other" ] || [ "${waw:-0}" -ne "${on128:-0}" ] || [ "${waw:-0}" -gt $(( ${n:-0} * frames )) ]; then
    echo "validation with shared border barriers reported more than those: $joined/validation.txt (write-after-write $waw, on the scratch buffer $on128, allowed ${n:-0} x $frames)"
    echo "$other" | grep -oE 'VUID-[A-Za-z0-9_-]+|SYNC-[A-Z0-9-]+' | sort | uniq -c | sort -rn | head -20
    exit 1
fi
echo "validation with shared border barriers: only the $waw write-after-writes of the ${n:-0} border passes a frame (x $frames frames) on the scratch buffer"
