#!/usr/bin/env python3
"""GPU time per frame, the port against AMD's DLL under Wine, through the harness's timing mode (TRACE_TIMING=1:
timestamps around prepare and generate in the one command buffer of a frame; the driver's own Vulkan functions given to
the provider, so the CPU times are its own).

    python3 -I tests/perf.py [-k substring ...] [--frames N] [--rounds R] [--only port|amd] [--cold] [--build DIR] [--re DIR]

Each case runs R rounds, alternating port and AMD so clock and heat drift hit both alike; the first 5 frames of each
run are dropped (first-use costs). Prints the median and the 10th/90th percentile of each over all rounds, and writes
<re>/perf/summary.json. --cold disables Mesa's shader cache (MESA_SHADER_CACHE_DISABLE) so create's time is the real
pipeline compile.
"""
import argparse, json, os, statistics, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WARMUP = 5

CASES = {   # name -> (W, H, RW, RH)
    '1920x1080_native': (1920, 1080, 1920, 1080),
    '1920x1080_s067': (1920, 1080, 1286, 724),
    '2560x1440_native': (2560, 1440, 2560, 1440),
    '2560x1440_s067': (2560, 1440, 1715, 965),
    '3840x2160_native': (3840, 2160, 3840, 2160),
    '3840x2160_s067': (3840, 2160, 2573, 1447),
}


def run_one(who, name, dims, frames, build, re_dir, cold):
    env = dict(os.environ, TRACE_TIMING='1', TRACE_CAMERA='0.5')
    if cold: env['MESA_SHADER_CACHE_DISABLE'] = 'true'
    args = [str(x) for x in dims] + [str(frames), '999999']
    if who == 'port':
        out = os.path.join(re_dir, 'perf', 'port', name)
        os.makedirs(out, exist_ok=True)
        cmd = [os.path.join(build, 'trace'), os.path.join(build, 'amd_fidelityfx_framegeneration_vk.so'), out] + args
    else:
        out = os.path.join(re_dir, 'perf', 'amd', name)
        cmd = [os.path.join(ROOT, 'harness', 'run_amd.sh'), name] + args
        env.update(FSR4FG_BUILD=build, FSR4FG_RE=os.path.join(re_dir, 'perf', 'amd'))
    r = subprocess.run(cmd, env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    path = os.path.join(out, 'timing.txt')
    if r.returncode or not os.path.exists(path):
        raise SystemExit('%s %s failed (%d):\n%s' % (who, name, r.returncode, r.stdout[-2000:]))
    create, rows = None, []
    for line in open(path):
        if line.startswith('# create_ms'): create = float(line.split()[2])
        elif not line.startswith('#'): rows.append([float(x) for x in line.split()[1:]])
    return create, rows[WARMUP:]


def stats(xs):
    xs = sorted(xs)
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return {'median': statistics.median(xs), 'p10': q(0.1), 'p90': q(0.9), 'n': len(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', action='append', default=[])
    ap.add_argument('--frames', type=int, default=65)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--only', choices=('port', 'amd'))
    ap.add_argument('--cold', action='store_true')
    ap.add_argument('--build', default=os.environ.get('FSR4FG_BUILD', os.path.join(ROOT, 'build')))
    ap.add_argument('--re', default=os.environ.get('FSR4FG_RE', os.path.join(ROOT, 're')))
    a = ap.parse_args()
    who_list = [a.only] if a.only else ['port', 'amd']
    summary = {}
    print('%-18s %-4s %22s %22s %10s %10s %9s' % ('case', 'who', 'prepare GPU ms', 'generate GPU ms', 'total', 'CPU ms', 'create'))
    for name, dims in CASES.items():
        if a.k and not any(k in name for k in a.k): continue
        acc = {w: {'rows': [], 'create': []} for w in who_list}
        for _ in range(a.rounds):
            for w in who_list:
                create, rows = run_one(w, name, dims, a.frames, a.build, a.re, a.cold)
                acc[w]['rows'] += rows
                acc[w]['create'].append(create)
        summary[name] = {}
        for w in who_list:
            rows = acc[w]['rows']
            s = {'prepare_gpu': stats([r[0] for r in rows]), 'generate_gpu': stats([r[1] for r in rows]),
                 'total_gpu': stats([r[0] + r[1] for r in rows]), 'cpu': stats([r[2] + r[3] for r in rows]),
                 'create_ms': statistics.median(acc[w]['create'])}
            summary[name][w] = s
            f = lambda st: '%.3f (%.3f-%.3f)' % (st['median'], st['p10'], st['p90'])
            print('%-18s %-4s %22s %22s %10.3f %10.3f %9.1f' % (name, w, f(s['prepare_gpu']), f(s['generate_gpu']), s['total_gpu']['median'],
                                                              s['cpu']['median'], s['create_ms']))
        if len(who_list) == 2:
            p, m = summary[name]['port']['total_gpu']['median'], summary[name]['amd']['total_gpu']['median']
            print('%-18s port/AMD total GPU: %.3f' % ('', p / m))
        sys.stdout.flush()
    os.makedirs(os.path.join(a.re, 'perf'), exist_ok=True)
    with open(os.path.join(a.re, 'perf', 'summary%s.json' % ('_cold' if a.cold else '')), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
