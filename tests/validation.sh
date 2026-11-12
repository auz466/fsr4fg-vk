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
export VK_KHRONOS_VALIDATION_SYNCVAL_FULL_VALIDATION=true
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
echo "validation clean ($(grep -c 'generate -> 0' "$out/stdout.txt") frames; AMD's cooperative-matrix shaders reported as known)"
