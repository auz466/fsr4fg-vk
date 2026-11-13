#!/bin/sh
# fault_sweep.sh [first last]: the port with its n-th Vulkan object creation failing (the harness's TRACE_FAIL_AT), for
# every n: it must not crash, must report the failure as an error code, and must leave none of its objects behind once
# its context is destroyed (or its create has failed). Prints one line per problem and a summary.
root=$(cd "$(dirname "$0")/.." && pwd); build=${FSR4FG_BUILD:-$root/build}; out=${FSR4FG_RE:-$root/re}/fault
first=${1:-1}; last=${2:-0}
mkdir -p "$out"
if [ "$last" = 0 ]; then
    TRACE_DUMP_ALL= "$build/trace" "$build/amd_fidelityfx_framegeneration_vk.so" "$out/n0" 1280 720 858 483 2 99 >/dev/null 2>&1
    last=$(grep -c '^Create\|^AllocateMemory\|^AllocateDescriptorSets' "$out/n0/trace.log")
fi
bad=0; n=$first
while [ "$n" -le "$last" ]; do
    rm -rf "$out/n"; TRACE_FAIL_AT=$n TRACE_DUMP_ALL= timeout 60 "$build/trace" "$build/amd_fidelityfx_framegeneration_vk.so" "$out/n" 1280 720 858 483 2 99 > "$out/n.txt" 2>&1
    rc=$?
    left=$(grep -o 'objects left after [a-z ]*: [0-9]*' "$out/n.txt" | grep -o '[0-9]*$')
    failed=$(grep -c ' -> [1-9]' "$out/n/trace.log" 2>/dev/null)
    if [ "$rc" -gt 2 ] || [ -z "$left" ] || [ "$left" != 0 ] || [ "$failed" = 0 ]; then
        echo "n=$n: rc=$rc objects left=${left:-?} failed calls=$failed  ($(grep -m1 'FAILING' "$out/n/trace.log" 2>/dev/null))"
        bad=$((bad + 1))
    fi
    n=$((n + 1))
done
echo "fault sweep $first..$last: $bad problems"
[ "$bad" = 0 ]
