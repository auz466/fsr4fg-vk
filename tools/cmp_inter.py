"""Compare intermediate images of an AMD run and a port run, same frame: cmp_inter.py amd mine frame 'name:amdid:mineid:dtype:ch' ..."""
import sys, numpy as np, re, os
a, b, fr = sys.argv[1], sys.argv[2], int(sys.argv[3])
def size(d, iid):
    for line in open(d + '/trace.log'):
        m = re.match(r'dump f%d_img%s_m0\.raw img\d+ mip0 (\d+)x(\d+)' % (fr, iid), line)
        if m: return int(m.group(1)), int(m.group(2))
for spec in sys.argv[4:]:
    name, ia, ib, dt, ch = spec.split(':'); ch = int(ch)
    w, h = size(a, ia)
    x = np.fromfile('%s/f%d_img%s_m0.raw' % (a, fr, ia), dt).reshape(h, w, ch)
    y = np.fromfile('%s/f%d_img%s_m0.raw' % (b, fr, ib), dt).reshape(h, w, ch)
    if dt in ('u4', 'u1'):
        ne = (x != y).any(-1); print('%-12s differing texels %.3f%% (%d)' % (name, 100 * ne.mean(), ne.sum()))
        if ne.any():
            yy, xx = np.argwhere(ne)[0]; print('   first at', xx, yy, 'amd', x[yy, xx], 'mine', y[yy, xx])
    else:
        x, y = x.astype(np.float64), y.astype(np.float64)
        d = np.abs(np.nan_to_num(x) - np.nan_to_num(y))
        print('%-12s max diff %.4g mean %.4g, differing %.3f%% | amd range %.4g..%.4g mine %.4g..%.4g' % (name, d.max(), d.mean(), 100 * (d.max(-1) > 1e-3).mean(), np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)))
