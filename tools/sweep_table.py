"""Across the size sweep: the tensor sizes table and each pass's group counts, one column per run."""
import sys, re, glob, os, struct, json
runs = {}
for d in sorted(glob.glob('re/sz_*')):
    m = re.match(r'sz_(\d+)x(\d+)_(\d+)x(\d+)', os.path.basename(d)); W, H, RW, RH = map(int, m.groups())
    cur = None; disp = []; head = None
    for line in open(d + '/trace.log'):
        mm = re.match(r'=== FRAME (\d+)', line)
        if mm: cur = int(mm.group(1)); continue
        if cur != 1: continue
        if line.startswith('DISPATCH'):
            g = re.search(r' (\d+)x(\d+)x(\d+) pipe', line); e = re.search(r'entry=(\S+)', line).group(1)
            head = {'entry': e, 'groups': tuple(map(int, g.groups())), 'ubo': {}}; disp.append(head)
        elif ' ubo ' in line and head is not None:
            mm = re.search(r'b(\d+)\[0\] ubo buf\d+ off=\d+ range=(\d+) data=(\S+)', line)
            head['ubo'][int(mm.group(1))] = bytes.fromhex(mm.group(3))[:int(mm.group(2))]
    runs[(W, H, RW, RH)] = disp
json.dump({'%d,%d,%d,%d' % k: [{'entry': x['entry'], 'groups': x['groups'], 'ubo': {str(b): v.hex() for b, v in x['ubo'].items()}} for x in v] for k, v in runs.items()}, open('re/sweep.json', 'w'))
keys = sorted(runs)
print('runs:', keys)
ts = {k: struct.unpack('<368I', next(x['ubo'][257] for x in runs[k] if 257 in x['ubo'])) for k in keys}
for k in keys:   # the same table on every pass of a run?
    assert all(x['ubo'][257] == next(y['ubo'][257] for y in runs[k] if 257 in y['ubo']) for x in runs[k] if 257 in x['ubo'])
print('tensor sizes (w,h) per index; z,w columns:', {ts[k][i * 4 + 2:i * 4 + 4] for k in keys for i in range(92)})
for i in range(92):
    print('%2d ' % i + '  '.join('%4dx%-4d' % (ts[k][i * 4], ts[k][i * 4 + 1]) for k in keys))
