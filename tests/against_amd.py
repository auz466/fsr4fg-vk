#!/usr/bin/env python3
"""The regression suite: every case through AMD's DLL under Wine and through the port, and any byte of difference in a
generated frame fails. The harness's own inputs (colour, depth, motion) are compared too: a difference there is the
harness's fault, not the port's, and also fails.

    python3 -I tests/against_amd.py [-k substring ...] [--list] [--refresh-amd] [--keep] [--build DIR] [--re DIR]

AMD's results are cached as SHA-256 hashes per image in <re>/suite/amd/<case>.json, keyed by the case, the harness
source, AMD's DLL and the rewritten shaders, so AMD only runs again when one of those changes (or with --refresh-amd).
The raw images are deleted after a passing case unless --keep. A failing case keeps the port's images in
<re>/suite/port/<case>/ (and AMD's in <re>/suite/amd/<case>/ if AMD ran this time; --refresh-amd -k <case> makes it
run), for tools/compare.py.
"""
import argparse, glob, hashlib, json, os, re, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INVERTED, INFINITE = 1 << 3, 1 << 4


def cases():
    """name -> (W, H, RW, RH, frames, environment)"""
    out = {}
    for w, h in ((1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), (5120, 2880), (1001, 777)):
        for scale_name, scale in (('native', 1.0), ('s067', 0.67)):
            rw, rh = (w, h) if scale == 1.0 else (round(w * scale), round(h * scale))
            for cam_name, cam in (('still', None), ('moving', '0.5')):
                env = {'TRACE_CAMERA': cam} if cam else {}
                out['%dx%d_%s_%s' % (w, h, scale_name, cam_name)] = (w, h, rw, rh, 4, env)
    out['1920x1080_s067_moving_reset3'] = (1920, 1080, 1286, 724, 6, {'TRACE_CAMERA': '0.5', 'TRACE_RESET_AT': '3'})
    for flags, nm in ((0, 'depth_std_finite'), (INVERTED, 'depth_inv_finite'), (INFINITE, 'depth_std_infinite'),
                      (INVERTED | INFINITE, 'depth_inv_infinite')):
        env = {'TRACE_CAMERA': '0.5', 'TRACE_CREATE_FLAGS': str(flags)}
        if not flags & INFINITE: env['TRACE_FAR'] = '1000'
        out['1920x1080_s067_moving_' + nm] = (1920, 1080, 1286, 724, 4, env)
    # (other fields of view and planes: the camera matrices use tanf and the planes in float)
    for fov, near, far, nm in (('0.7', '0.05', '500', 'fov07_finite'), ('1.3', '0.3', None, 'fov13_infinite'), ('0.42', '1', '20000', 'fov042_finite'),
                                 ('0.42', '1', '20000', 'fov042_std_finite')):
        env = {'TRACE_CAMERA': '0.5', 'TRACE_FOV': fov, 'TRACE_NEAR': near}
        if far: env.update(TRACE_FAR=far, TRACE_CREATE_FLAGS=str(0 if 'std' in nm else INVERTED))
        out['2560x1440_s067_moving_' + nm] = (2560, 1440, 1715, 965, 4, env)
    return out


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''): h.update(chunk)
    return h.hexdigest()


def amd_key(build, case):
    h = hashlib.sha256(json.dumps(case, sort_keys=True).encode())
    h.update(open(os.path.join(ROOT, 'harness', 'trace.cpp'), 'rb').read())
    h.update(sha_file(os.path.join(build, 'wine', 'amd_fidelityfx_framegeneration_vk.dll')).encode())
    for f in sorted(glob.glob(os.path.join(build, 'gen', 'patched', '*.spv'))):
        h.update(os.path.basename(f).encode() + sha_file(f).encode())
    return h.hexdigest()


def images(run_dir, frames, gen):
    """(frame, name) -> hash, for the harness's inputs and the generated frames; and the run's per-frame results"""
    log = open(os.path.join(run_dir, 'trace.log'), newline='').read()
    wanted = {'in_color', 'in_depth', 'in_motion'} | {'out_made' + (str(k) if k else '') for k in range(gen)}
    out = {}
    for m in re.finditer(r'^dump (f(\d+)_img\d+_m0\.raw) img\d+ mip0 (\d+)x(\d+) format=(\d+) name=(\S+)\r?$', log, re.M):
        if m.group(6) in wanted:
            path = os.path.join(run_dir, m.group(1))
            out['f%s %s %sx%s fmt%s' % (m.group(2), m.group(6), m.group(3), m.group(4), m.group(5))] = sha_file(path)
            if m.group(6).startswith('out_made') and not any(open(path, 'rb').read()):
                out['f%s %s' % (m.group(2), m.group(6))] = 'ALL ZERO'   # (nothing written: never a pass, even if both agree)
    return out


def run(cmd, env, cwd, log):
    t = time.time()
    with open(log, 'w') as f:
        r = subprocess.run(cmd, env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t


def results_of(stdout_log, frames):
    """every 'frame N: generate -> R' line: the run must have made every frame with result 0"""
    got = re.findall(r'frame (\d+): generate -> (\d+)', open(stdout_log, errors='replace').read())
    return len(got) == frames and all(r == '0' for _, r in got), got


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-k', action='append', default=[], help='only cases whose name contains this (repeatable)')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--refresh-amd', action='store_true', help="run AMD's DLL again even if cached")
    ap.add_argument('--keep', action='store_true', help='keep the raw images of passing cases')
    ap.add_argument('--build', default=os.environ.get('FSR4FG_BUILD', os.path.join(ROOT, 'build')))
    ap.add_argument('--re', default=os.environ.get('FSR4FG_RE', os.path.join(ROOT, 're')))
    a = ap.parse_args()
    build, suite = os.path.abspath(a.build), os.path.join(os.path.abspath(a.re), 'suite')
    todo = {n: c for n, c in cases().items() if not a.k or any(k in n for k in a.k)}
    if a.list:
        for n, c in todo.items(): print(n, *c)
        return 0
    for f in ('amd_fidelityfx_framegeneration_vk.so', 'trace', 'wine/trace.exe', 'wine/amd_fidelityfx_framegeneration_vk.dll'):
        if not os.path.exists(os.path.join(build, f)): raise SystemExit('missing %s/%s: build with mingw-w64 and -DFSR4FG_AMD_DLL' % (build, f))
    os.makedirs(os.path.join(suite, 'amd'), exist_ok=True), os.makedirs(os.path.join(suite, 'port'), exist_ok=True)
    failed = []
    for name, case in todo.items():
        w, h, rw, rh, frames, cenv = case
        gen = int(cenv.get('TRACE_GEN', '1'))
        args = [str(x) for x in (w, h, rw, rh, frames, 0)]
        env = {k: v for k, v in os.environ.items() if not k.startswith('TRACE_')}
        env.update(cenv, TRACE_DUMP_ALL='')
        # AMD, from the cache when it can be
        key = amd_key(build, case)
        cache = os.path.join(suite, 'amd', name + '.json')
        amd = json.load(open(cache)) if os.path.exists(cache) and not a.refresh_amd else None
        amd_dir, amd_time = os.path.join(suite, 'amd', name), 0.0
        if not amd or amd.get('key') != key:
            rc, amd_time = run([os.path.join(ROOT, 'harness', 'run_amd.sh'), name] + args,
                               dict(env, FSR4FG_BUILD=build, FSR4FG_RE=os.path.join(suite, 'amd')), ROOT, os.path.join(suite, 'amd', name + '.out'))
            ok, got = results_of(os.path.join(suite, 'amd', name + '.out'), frames)
            if rc or not ok:
                print('%-40s AMD RUN FAILED (rc %d, frames %s): see %s.out' % (name, rc, got, os.path.join(suite, 'amd', name)))
                failed.append(name)
                continue
            amd = {'key': key, 'case': case, 'images': images(amd_dir, frames, gen)}
            json.dump(amd, open(cache, 'w'), indent=1)
        # the port, every time
        port_dir = os.path.join(suite, 'port', name)
        shutil.rmtree(port_dir, ignore_errors=True)
        rc, port_time = run([os.path.join(build, 'trace'), os.path.join(build, 'amd_fidelityfx_framegeneration_vk.so'), port_dir] + args,
                            env, ROOT, port_dir + '.out')
        ok, got = results_of(port_dir + '.out', frames)
        mine = images(port_dir, frames, gen) if os.path.exists(os.path.join(port_dir, 'trace.log')) else {}
        bad = sorted(k for k in set(amd['images']) | set(mine) if amd['images'].get(k) != mine.get(k) or mine.get(k) == 'ALL ZERO')
        made = sum(1 for k in amd['images'] if ' out_made' in k)
        if rc or not ok or bad or made != frames * gen:
            failed.append(name)
            harness = [k for k in bad if ' in_' in k]
            print('%-40s FAIL  rc %d, frames %s, %d of %d images differ%s' % (name, rc, got, len(bad), len(amd['images']),
                  ' (harness inputs differ: %s)' % harness[:3] if harness else ''))
            for k in bad[:6]: print('      %-36s amd %s  port %s' % (k, str(amd['images'].get(k))[:12], str(mine.get(k))[:12]))
            if made != frames * gen: print('      AMD made %d frames, expected %d' % (made, frames * gen))
        else:
            print('%-40s pass  %d frames bit-identical (+%d inputs)  [amd %s, port %.1fs]' % (
                name, made, len(amd['images']) - made, 'cached' if not amd_time else '%.1fs' % amd_time, port_time))
            for d in (port_dir, amd_dir):
                if not a.keep: shutil.rmtree(d, ignore_errors=True)
    # cases meant to differ must make different frames in AMD's DLL, or they test nothing (the moving camera used to
    # make the same frames as the still one: AMD's forward motion pass ignores the camera where the motion vectors do
    # not agree with it)
    def made(n):
        f = os.path.join(suite, 'amd', n + '.json')
        return {k: v for k, v in json.load(open(f))['images'].items() if ' out_made' in k} if os.path.exists(f) else None
    base = '1920x1080_s067_moving'
    groups = [['%dx%d_%s_%s' % (w, h, s, c) for c in ('still', 'moving')] for w, h in ((1280, 720), (1920, 1080), (5120, 2880))
              for s in ('native', 's067')]
    groups += [[base, base + '_reset3'], [base + '_depth_' + m for m in ('std_finite', 'inv_finite', 'std_infinite', 'inv_infinite')]]
    for g in groups:
        if not all(n in todo for n in g): continue
        frames = [made(n) for n in g]
        if any(f is None for f in frames): continue
        same = [(g[i], g[j]) for i in range(len(g)) for j in range(i + 1, len(g)) if frames[i] == frames[j]]
        if same:
            print('NOT SENSITIVE: AMD made the same frames for %s' % ', '.join('%s = %s' % p for p in same))
            failed.append('sensitivity')
    print('\n%d of %d cases passed' % (len(todo) - len([f for f in failed if f != 'sensitivity']), len(todo)))
    if failed: print('failed:', ' '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
