#!/bin/sh
# validation.sh [W H RW RH frames]: the port under VK_LAYER_KHRONOS_validation, with synchronisation validation, on
# the moving camera. Fails if the layer reports anything. Needs the Khronos validation layer (vulkan-validation-layers).
#   FSR4FG_BUILD   the build directory (default: <repo>/build)
#   FSR4FG_RE      where the run goes (default: <repo>/re), as validation/
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
build=${FSR4FG_BUILD:-$root/build}; out=${FSR4FG_RE:-$root/re}/validation
rm -rf "$out"; mkdir -p "$out"
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_KHRONOS_VALIDATION_VALIDATE_CORE=true
export VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true
export VK_KHRONOS_VALIDATION_SYNCVAL_SUBMIT_TIME_VALIDATION=true
export VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC=true
export VK_KHRONOS_VALIDATION_VALIDATE_BEST_PRACTICES=false
export VK_KHRONOS_VALIDATION_REPORT_FLAGS=error,warn
export VK_KHRONOS_VALIDATION_DEBUG_ACTION=VK_DBG_LAYER_ACTION_LOG_MSG
export VK_KHRONOS_VALIDATION_LOG_FILENAME="$out/validation.txt"
export VK_KHRONOS_VALIDATION_DUPLICATE_MESSAGE_LIMIT=3
TRACE_CAMERA=${TRACE_CAMERA-0.5} TRACE_RESET_AT=${TRACE_RESET_AT-2} TRACE_DUMP_ALL= \
    "$build/trace" "$build/amd_fidelityfx_framegeneration_vk.so" "$out" "${1:-1920}" "${2:-1080}" "${3:-1286}" "${4:-724}" "${5:-4}" 99 \
    > "$out/stdout.txt" 2>&1 || { echo "the harness failed: $out/stdout.txt"; exit 1; }
# (with VK_INSTANCE_LAYERS naming a layer that is missing, instance creation fails: a clean run did load it)
if [ -s "$out/validation.txt" ]; then
    echo "validation reported $(grep -c 'VUID\|SYNC-' "$out/validation.txt") problems: $out/validation.txt"
    grep -o 'VUID-[A-Za-z0-9_-]*\|SYNC-[A-Z0-9-]*' "$out/validation.txt" | sort | uniq -c | sort -rn | head -20
    exit 1
fi
echo "validation clean ($(grep -c 'generate -> 0' "$out/stdout.txt") frames)"
