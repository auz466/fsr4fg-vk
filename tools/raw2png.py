"""Raw texel dumps from harness/trace.cpp as PNGs: raw2png.py <trace dir> <prefix> [out dir]. Formats by VkFormat number."""
import sys, re, os, numpy as np
from PIL import Image
d, prefix = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(d, 'png')
os.makedirs(out, exist_ok=True)
FMT = {37: ('u1', 4), 44: ('u1', 4), 83: ('f2', 2), 100: ('f4', 1), 101: ('f4', 2), 97: ('f2', 4), 76: ('f2', 1), 9: ('u1', 1), 16: ('u1', 2),
       110: ('u8', 1), 98: ('u4', 1), 70: ('u2', 1), 77: ('u2', 2), 81: ('u2', 2), 103: ('f4', 2), 64: ('u4', 1), 74: ('u2', 1), 13:('u1',1)}
for line in open(os.path.join(d, 'trace.log')):
    m = re.match(r'dump (%s_img(\d+)_m(\d+)\.raw) img\d+ mip\d+ (\d+)x(\d+) format=(\d+) name=(.*)' % re.escape(prefix), line)
    if not m: continue
    f, iid, mip, w, h, fmt, name = m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5)), int(m.group(6)), m.group(7).strip()
    if fmt not in FMT: print('skip format', fmt, f); continue
    dt, n = FMT[fmt]
    a = np.fromfile(os.path.join(d, f), dtype=dt).reshape(h, w, n)
    info = '%s %dx%d fmt%d' % (f, w, h, fmt)
    if fmt == 64:   # A2B10G10R10
        v = a[..., 0]; a = np.stack([(v & 1023), (v >> 10) & 1023, (v >> 20) & 1023], -1) / 1023.0; dt = 'f4'
    if dt == 'u1' and n == 4: img = a[..., :3]
    elif dt == 'u8':
        v = a[..., 0]; print(info, 'u64 nonzero=%d' % np.count_nonzero(v), 'distinct', len(np.unique(v)))
        img = np.stack([((v >> s) & 0xffff).astype(np.float32) for s in (0, 16, 32)], -1); img = (255 * img / max(img.max(), 1)).astype(np.uint8)
    else:
        a = a.astype(np.float32); fin = a[np.isfinite(a)]
        print(info, 'min %.5g max %.5g mean %.5g' % (fin.min(), fin.max(), fin.mean()) if fin.size else 'no finite', name)
        if n == 1: lo, hi = np.percentile(fin, [1, 99]) if fin.size else (0, 1); img = np.repeat(np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1), 3, -1)
        elif n == 2:
            s = max(np.percentile(np.abs(fin), 99), 1e-9) if fin.size else 1
            img = np.concatenate([np.clip(a / s * 0.5 + 0.5, 0, 1), np.full((h, w, 1), 0.5, np.float32)], -1)
        else: img = np.clip(a[..., :3], 0, 1)
        img = (np.nan_to_num(img) * 255).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(img)).save(os.path.join(out, '%s_img%s_m%s%s.png' % (prefix, iid, mip, '_' + name if name else '')))
