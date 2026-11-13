#!/usr/bin/env python3
"""Which of the network's border passes may run with no barrier after the pass before them.

    python3 -I tools/probe_borders.py [--build DIR] [--out DIR] [--sizes WxH,...] [--write]

A border pass binds only the scratch buffer and the tensor sizes, loads nothing from the scratch buffer and stores only
zeros to it (checked here in its SPIR-V). It fills the border of a tensor that the next work pass reads. If what it
writes is disjoint from what the work pass before it (c) writes, and c's output does not depend on those words, the two
can share one barrier: nothing they do can be ordered differently. The harness's TRACE_PROBE measures exactly that,
per pair (see harness/trace.cpp); this runs it at each size for both networks (4k below 3840x2160 inclusive, 8k above)
and prints the pairs.

That c zeroes what it writes out of range holds at most sizes but not all (pass 189 after 188, at 1088x1088 only), so
nothing is assumed of a size that was not probed: with --write, data/model.json gets the border passes
("border_passes") and, for each display size probed, the ones that keep their barrier there ("border_barriers"); a
size probed must pass at both render scales. setup.py turns that into tables; the library skips a barrier only at a
display size in the table, and records every barrier at any other.
"""
import argparse, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (the display sizes of common screens and windows: 16:9, 16:10, 21:9, 32:9, 4:3, 3:2, handhelds)
SIZES = ['1024x576', '1152x648', '1280x720', '1366x768', '1600x900', '1920x1080', '2048x1152', '2560x1440', '2880x1620', '3200x1800',
         '3840x2160', '5120x2880', '7680x4320', '1280x800', '1440x900', '1680x1050', '1920x1200', '2560x1600', '2880x1800', '3840x2400',
         '2560x1080', '3440x1440', '3840x1600', '5120x2160', '3840x1080', '5120x1440', '7680x2160', '800x600', '1024x768', '1280x960',
         '1600x1200', '1280x1024', '2256x1504', '3000x2000', '2160x1440', '1920x1280', '2880x1920', '2400x1080', '1360x768', '1536x864',
         '1600x1024', '3072x1728', '6016x3384']


def zero_passes(build):
    """entry -> True for the passes whose shader only stores zeros to the scratch buffer and loads nothing from it."""
    gen = open(os.path.join(build, 'gen', 'assets_gen.h')).read()
    blob = open(os.path.join(build, 'gen', 'assets.bin'), 'rb').read()
    out = {}
    tmp = os.path.join(build, 'gen', 'probe_tmp.spv')
    for variant in ('4K', '8K'):
        body = re.search(r'GEN_PASSES_%s\[\]\s*=\s*\{(.*?)\n\};' % variant, gen, re.S).group(1)
        for line in body.split('\n'):
            m = re.match(r'\s*\{(\d+), (\d+), "(\w+)".*\}\}, (\d+), \{(.*)\}\},', line)
            if not m: continue
            off, size, entry, nb, binds = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)), m.group(5)
            if nb != 2 or 'RES_SCRATCH' not in binds or 'RES_UBO_TENSOR' not in binds: continue
            open(tmp, 'wb').write(blob[off:off + size])
            dis = subprocess.run(['spirv-dis', '--raw-id', tmp], capture_output=True, text=True, check=True).stdout
            sb = re.findall(r'%(\d+) = OpVariable %\d+ StorageBuffer', dis)
            if len(sb) != 1: continue
            ptrs = set(re.findall(r'%(\d+) = OpAccessChain %\d+ %' + sb[0] + ' ', dis))
            zeros = set(re.findall(r'%(\d+) = OpConstant %\d+ 0\n', dis))
            loads = [p for p in re.findall(r'OpLoad %\d+ %(\d+)', dis) if p in ptrs]
            stored = {v for p, v in re.findall(r'OpStore %(\d+) %(\d+)', dis) if p in ptrs}
            ok = not loads and stored and stored <= zeros and 'OpAtomic' not in dis and 'OpCopyMemory' not in dis
            out[entry] = bool(ok)
    os.remove(tmp)
    return out


def probe(a, zero, size, rw, rh):
    w, h = (int(x) for x in size.split('x'))
    out = os.path.join(a.out, '%s_%dx%d' % (size, rw, rh))
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, TRACE_PROBE='1', TRACE_PROBE_ZERO=','.join(zero), TRACE_CAMERA='0.5')
    cmd = [os.path.join(a.build, 'trace'), os.path.join(a.build, 'amd_fidelityfx_framegeneration_vk.so'), out, str(w), str(h), str(rw), str(rh), '3', '99']
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode: raise SystemExit('probe at %s failed:\n%s' % (size, r.stderr[-3000:]))
    ok_of = {}
    for m in re.finditer(r'PROBE (\w+) after (\w+): wz=(\d+) wc=(\d+) waw=\d+ waw_nonzero=(\d+) infl_real=(\d+) infl_random=(\d+) notzero=(\d+)', r.stderr):
        entry, c = m.group(1), m.group(2)
        wz, wc, wawnz, ir, ix, nz = (int(x) for x in m.groups()[2:])
        ok = wz > 0 and wc > 0 and wawnz == 0 and ir == 0 and ix == 0 and nz == 0
        ok_of[entry] = ok
        if not ok: print('  %s (render %dx%d): %s after %s keeps its barrier (wz=%d wc=%d waw_nonzero=%d infl %d/%d notzero=%d)' %
                         (size, rw, rh, entry, c, wz, wc, wawnz, ir, ix, nz))
    return ok_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', default=os.path.join(ROOT, 'build'))
    ap.add_argument('--out', default=os.path.join(ROOT, 're', 'probe'))
    ap.add_argument('--sizes', default=','.join(SIZES))
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    zp = zero_passes(a.build)
    zero = [e for e, ok in zp.items() if ok]
    print('border passes (zeros only): %d of %d two-binding passes' % (len(zero), len(zp)))
    keep = {}
    for size in a.sizes.split(','):
        w, h = (int(x) for x in size.split('x'))
        variant = '8k' if w > 3840 or h > 2160 else '4k'
        mine = [e for e in zero if e.startswith('mlfi_%s_' % variant)]
        native, scaled = probe(a, zero, size, w, h), probe(a, zero, size, round(w * 0.67), round(h * 0.67))
        keep[size] = [e for e in mine if not (native.get(e) and scaled.get(e))]   # (not probed counts as not independent)
        print('%s: %d of %d border passes need no barrier of their own' % (size, len(mine) - len(keep[size]), len(mine)))
        sys.stdout.flush()
    if a.write:
        path = os.path.join(ROOT, 'data', 'model.json')
        model = json.load(open(path))
        model['border_passes'] = zero
        model['border_barriers'] = keep
        with open(path, 'w') as f:
            json.dump(model, f, indent=0)
        print('written to', path)


if __name__ == '__main__':
    main()
