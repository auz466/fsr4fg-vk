"""Summarise a trace.log: per frame, each dispatch on one line (entry, groups, bindings), UBO contents listed once each."""
import sys, re, collections
log = sys.argv[1]; frame_want = int(sys.argv[2]) if len(sys.argv) > 2 else None
frames = collections.OrderedDict(); cur = None; d = None; other = []
for line in open(log):
    line = line.rstrip('\n')
    m = re.match(r'=== FRAME (\d+)', line)
    if m: cur = frames.setdefault(int(m.group(1)), []); continue
    if cur is None: continue
    if line.startswith('DISPATCH'):
        d = {'head': line, 'binds': []}; cur.append(d)
    elif line.startswith('  s') and d is not None: d['binds'].append(line.strip())
    elif line.startswith('==='): cur.append({'head': line, 'binds': []}); d = None
    else: cur.append({'head': '    ' + line, 'binds': []}); d = None
for fi, items in frames.items():
    if frame_want is not None and fi != frame_want: continue
    print('######## FRAME', fi)
    for it in items:
        h = it['head']
        m = re.match(r'DISPATCH (\d+) (\w+) (\S+) pipe\d+ mod\d+ entry=(\S+)', h)
        if not m: print(h); continue
        parts = []
        for b in it['binds']:
            mm = re.match(r's\d+ b(\d+)\[(\d+)\] (\w+)(?:\(t\d+\))? (.*)', b)
            bi, el, kind, rest = mm.groups()
            if kind == 'ubo':
                r = re.match(r'buf\d+ off=\d+ range=(\d+) data=(\S+)', rest)
                parts.append('b%s:ubo[%s]=%s' % (bi, r.group(1), r.group(2)[:int(r.group(1))*2] if len(sys.argv) > 3 else '..'))
            elif kind == 'sampler': parts.append('b%s:%s' % (bi, rest))
            elif kind in ('sampled', 'storage'):
                r = re.match(r'(img\d+) mip(\d+) layout=(\d+)\s*(.*)', rest)
                parts.append('b%s:%s%s%s%s' % (bi, 'W ' if kind == 'storage' else '', r.group(1), ('.m' + r.group(2)) if r.group(2) != '0' else '', (' ' + r.group(4)) if r.group(4) else ''))
            else:
                r = re.match(r'(buf\d+) off=(\d+) range=(-?\d+)', rest)
                parts.append('b%s:%s+%s' % (bi, r.group(1), r.group(2)))
        print('%s %-18s %-12s %s' % (m.group(1), m.group(4), m.group(3), '  '.join(parts)))
