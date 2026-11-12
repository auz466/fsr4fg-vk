"""Reflect each carved module: entry point, stage, local size, capabilities, and every bound variable (set, binding, kind, name)."""
import sys, struct, json, os, glob
src = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(src, '*.spv'))):
    d = open(f, 'rb').read(); w = struct.unpack('<%dI' % (len(d) // 4), d)
    names, deco, types, vars_, entry, caps, exts, lsize, consts, members = {}, {}, {}, [], None, [], [], None, {}, {}
    def s(ws):
        b = struct.pack('<%dI' % len(ws), *ws); return b[:b.index(b'\0')].decode('utf8', 'replace')
    i = 5
    while i < len(w):
        op, wc = w[i] & 0xffff, w[i] >> 16; a = w[i + 1:i + wc]
        if op == 17: caps.append(a[0])
        elif op == 10: exts.append(s(a))
        elif op == 15: entry = (a[0], s(a[2:]))
        elif op == 16 and a[1] == 17: lsize = a[2:5]
        elif op == 5: names[a[0]] = s(a[1:])
        elif op == 6: members.setdefault(a[0], {})[a[1]] = s(a[2:])
        elif op == 71: deco.setdefault(a[0], {})[a[1]] = a[2] if len(a) > 2 else True
        elif op == 72: deco.setdefault(('m', a[0], a[1]), {})[a[2]] = a[3] if len(a) > 3 else True
        elif op in (19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 4456): types[a[0]] = (op, a[1:])
        elif op == 43: consts[a[1]] = a[2]
        elif op == 59 and a[2] in (0, 2, 9, 11, 12): vars_.append((a[1], a[0], a[2]))
        i += wc
    def kind(tid, sc):
        op, a = types.get(tid, (0, ()))
        if op == 32: return kind(a[1], a[0])
        if op in (28, 29): return kind(a[0], sc) + '[]'
        if op == 25: return ('image%s' % ('RW' if a[5] == 2 else '')) + ':dim%d:fmt%d' % (a[1], a[6])
        if op == 26: return 'sampler'
        if op == 27: return 'combined'
        if op == 30:
            d_ = deco.get(tid, {})
            return 'ubo' if (2 in d_ and sc == 2) else ('pushconst' if sc == 9 else 'ssbo')
        return 'op%d' % op
    bind = []
    for vid, tid, sc in vars_:
        dd = deco.get(vid, {})
        if 33 in dd or sc == 9:
            bind.append({'set': dd.get(34), 'binding': dd.get(33), 'kind': kind(tid, sc), 'name': names.get(vid, '?')})
    bind.sort(key=lambda b: (b['set'] or 0, b['binding'] if b['binding'] is not None else -1))
    rows.append({'file': os.path.basename(f), 'size': len(d), 'entry': entry[1] if entry else None, 'model': None, 'local': lsize, 'caps': caps, 'exts': exts, 'bind': bind})
json.dump(rows, open(sys.argv[2], 'w'), indent=1)
import collections, re
g = collections.Counter(re.sub(r'\d+$', '#', r['entry'] or '?') for r in rows)
for k, v in sorted(g.items()): print(v, k)
