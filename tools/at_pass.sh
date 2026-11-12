#!/bin/sh
# at_pass.sh <pass number>: both providers' scratch buffers after that pass of the first frame, compared
cd "$(dirname "$0")/.."
n=$1; e=mlfi_4k_pass$n
TRACE_STOP_ENTRY=$e TRACE_DUMP_BUFFERS=100000000 harness/run_amd.sh at_amd 1920 1080 1920 1080 1 0 >/dev/null 2>&1
rm -rf re/at_mine; mkdir -p re/at_mine
TRACE_STOP_ENTRY=$e TRACE_DUMP_BUFFERS=100000000 TRACE_DUMP_ALL=1 ./build/trace build/amd_fidelityfx_framegeneration_vk.so re/at_mine 1920 1080 1920 1080 1 0 >/dev/null 2>&1
python3 -I - "$e" <<'PY'
import sys, re, numpy as np
def scratch(d):
    for line in open(d + '/trace.log'):
        m = re.match(r'dump (f0_buf\d+\.raw) buf\d+ size=58397248', line)
        if m: return np.fromfile(d + '/' + m.group(1), np.uint16)
a, b = scratch('re/at_amd'), scratch('re/at_mine')
fa, fb = a.view(np.float16).astype(np.float32), b.view(np.float16).astype(np.float32)
ne = a != b
print('%s: scratch halves differing %d of %d; NaN amd %d mine %d' % (sys.argv[1], ne.sum(), a.size, np.isnan(fa).sum(), np.isnan(fb).sum()))
if ne.any():
    idx = np.nonzero(ne)[0]; print('   first differing byte offset %d, last %d; e.g. amd %s mine %s' % (idx[0] * 2, idx[-1] * 2, fa[idx[:4]], fb[idx[:4]]))
PY
