"""Fit every pass's dispatch group counts to the tensor size table, from the size sweep (re/sweep.json).

Result: re/groups.json, per variant ('4k', '8k') a list per pass of three terms, each one of
  ['k', n]                         a constant
  ['t', index, comp, div, mul]     ceil(T[index][comp] / div) * mul
  ['pad', index, maxw, maxh, cw, ch]   the border-clearing passes: ceil(count / 32), count as their shader computes it
"""
import json, struct, sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spv import Module

sw = json.load(open('re/sweep.json'))
cat = {r['entry']: r for r in json.load(open('re/catalog.json'))}
runs = {}
for k, disp in sw.items():
    T = struct.unpack('<368I', bytes.fromhex(next(x['ubo']['257'] for x in disp if '257' in x['ubo'])))
    T = [(T[i * 4], T[i * 4 + 1]) for i in range(92)]
    variant = '8k' if '8k' in next(x['entry'] for x in disp if 'mlfi' in x['entry']) else '4k'
    runs[k] = (variant, T, [x for x in disp if x['entry'].startswith('mlfi')])


def pad_constants(entry):
    """(tensor index, max w, max h, clamp w, clamp h) of a border-clearing pass, read from its SPIR-V; None if it is not one"""
    m = Module(open('re/spv/' + cat[entry]['file'], 'rb').read())
    consts, comps, ints = {}, {}, set()
    for x in m.ins:
        if x.op == 21: ints.add(x.w[0])
        if x.op == 43 and x.w[0] in ints: consts[x.w[1]] = x.w[2]
        if x.op == 44: comps[x.w[1]] = x.w[2:]
    f = m.first_function()
    body = m.ins[f:]
    idx = None
    for n, x in enumerate(body):
        if x.op == 65 and len(x.w) == 5 and idx is None: idx = consts.get(x.w[4])   # OpAccessChain CsTensorSizes 0 N
        if x.op == 130 and x.w[2] in comps:   # OpISub v2uint max size
            mx = [consts[c] for c in comps[x.w[2]]]
            nxt = body[n + 1]
            if nxt.op == 12 and nxt.w[3] == 38:   # OpExtInst UMin
                cl = [consts[c] for c in comps[nxt.w[5] if nxt.w[4] == x.w[1] else nxt.w[4]]]
                return idx, mx[0], mx[1], cl[0], cl[1]
    return None


def pad_count(T, p):
    idx, mw, mh, cw, ch = p
    w, h = T[idx]
    dx, dy = min(mw - w, cw), min(mh - h, ch)
    w1 = 1 + w + dx
    return -(-(w1 + h + h * dx + w1 * dy) // 32)


out = {}
for variant in ('4k', '8k'):
    mine = [r for r in runs.values() if r[0] == variant]
    n = len(mine[0][2]); passes = []
    for i in range(n):
        entry = mine[0][2][i]['entry']
        terms = []
        obs = [[r[2][i]['groups'][d] for r in mine] for d in range(3)]
        pad = None
        for d in range(3):
            o = obs[d]
            if len(set(o)) == 1 and (len(mine) > 1 or d > 0 or o[0] in (1, 5190, 326)) and not (d == 0 and obs[1] == [1] * len(o) and obs[2] == [1] * len(o) and o[0] not in (5190, 326, 1)):
                terms.append(['k', o[0]]); continue
            if d == 0 and all(x == 1 for x in obs[1]) and all(x == 1 for x in obs[2]):
                pad = pad_constants(entry)
                if pad and all(pad_count(r[1], pad) == r[2][i]['groups'][0] for r in mine):
                    terms.append(['pad'] + list(pad)); continue
                pad = None
            found = None
            for t in range(92):
                for c in (0, 1):
                    for div in (1, 2, 4, 8, 16, 32, 64):
                        for mul in (1, 2, 3, 4, 6, 8):
                            if all(-(-r[1][t][c] // div) * mul == r[2][i]['groups'][d] for r in mine):
                                found = found or ['t', t, c, div, mul]
            if not found: raise SystemExit('no fit: %s dim %d %s' % (entry, d, o))
            terms.append(found)
        passes.append({'entry': entry, 'groups': terms})
    out[variant] = passes
    kinds = {}
    for p in passes: kinds[tuple(t[0] for t in p['groups'])] = kinds.get(tuple(t[0] for t in p['groups']), 0) + 1
    print(variant, len(passes), 'passes fitted over', len(mine), 'runs:', kinds)
json.dump(out, open('re/groups.json', 'w'), indent=0)
for p in out['4k'][:6] + out['4k'][150:153] + out['4k'][188:]: print(p)
print(out['8k'][1], out['8k'][153])
