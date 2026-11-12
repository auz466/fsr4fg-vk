"""Decode the UBOs of chosen dispatches in a trace as float / int / uint columns: ubo_dump.py trace.log frame [entry regex]"""
import sys, re, struct
log, frame, pat = sys.argv[1], int(sys.argv[2]), re.compile(sys.argv[3] if len(sys.argv) > 3 else 'main')
cur = None; head = None; seen = set()
for line in open(log):
    m = re.match(r'=== FRAME (\d+)', line)
    if m: cur = int(m.group(1)); continue
    if cur != frame: continue
    if line.startswith('DISPATCH'):
        e = re.search(r'entry=(\S+)', line).group(1); n = line.split()[1]; head = (n, e) if pat.search(e) else None; continue
    if line.startswith('LABEL begin'): label = line.split('begin ')[1].strip()
    if head and ' ubo ' in line:
        m = re.search(r'b(\d+)\[0\] ubo buf\d+ off=\d+ range=(\d+) data=(\S+)', line)
        b, rng, data = int(m.group(1)), int(m.group(2)), bytes.fromhex(m.group(3))[:int(m.group(2))]
        key = (head[1], b, data)
        if key in seen: continue
        seen.add(key)
        print('--- dispatch %s %s (%s) binding %d, %d bytes' % (head[0], head[1], label, b, rng))
        for o in range(0, len(data) - 3, 4):
            f, i, u = struct.unpack_from('<f', data, o)[0], struct.unpack_from('<i', data, o)[0], struct.unpack_from('<I', data, o)[0]
            print('  +%3d  %14.7g  %12d  0x%08x' % (o, f, i, u))
