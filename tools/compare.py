"""Compare the made frames of two harness runs (the same scene): compare.py <run a> <run b> [frame ...]"""
import sys, re, os, numpy as np
from PIL import Image
def dumps(d):
    out = {}
    for line in open(os.path.join(d, 'trace.log')):
        m = re.match(r'dump (f(\d+)_img\d+_m0\.raw) img\d+ mip0 (\d+)x(\d+) format=(\d+) name=(\S+)', line)
        if m: out[(int(m.group(2)), m.group(6))] = (os.path.join(d, m.group(1)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
    return out
a, b = dumps(sys.argv[1]), dumps(sys.argv[2])
frames = [int(x) for x in sys.argv[3:]] or sorted({k[0] for k in a if k[1] == 'out_made'} & {k[0] for k in b if k[1] == 'out_made'})
def load(e):
    f, w, h, fmt = e
    if fmt == 37: return np.fromfile(f, np.uint8).reshape(h, w, 4)[..., :3].astype(np.float32)
    if fmt == 64:
        v = np.fromfile(f, np.uint32).reshape(h, w); return np.stack([v & 1023, (v >> 10) & 1023, (v >> 20) & 1023], -1).astype(np.float32) / 1023 * 255
    if fmt == 97: return np.clip(np.fromfile(f, np.float16).reshape(h, w, 4)[..., :3].astype(np.float32), 0, 1) * 255
    raise SystemExit('format %d' % fmt)
worst = 0
for fr in frames:
    x, y = load(a[(fr, 'out_made')]), load(b[(fr, 'out_made')])
    cur = load(a[(fr, 'in_color')])
    d = np.abs(x - y); mse = float((d ** 2).mean())
    psnr = 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    same_as_input = float(np.abs(y - cur).mean())
    print('frame %d: max diff %.0f, mean diff %.4f, differing pixels %.3f%%, PSNR %.1f dB | b vs current input mean diff %.2f' %
          (fr, d.max(), d.mean(), 100 * (d.max(-1) > 2).mean(), psnr, same_as_input))
    worst = max(worst, d.mean())
    Image.fromarray(np.concatenate([x, y, np.clip(d * 8, 0, 255)], 1).astype(np.uint8)).save(os.path.join(sys.argv[2], 'compare_f%d.png' % fr))
sys.exit(0 if worst < 0.5 else 1)
