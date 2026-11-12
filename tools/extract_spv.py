"""Carve every SPIR-V module out of AMD's frame generation DLL (read as data; nothing is executed)."""
import sys, struct, hashlib, json, os
dll, out = sys.argv[1], sys.argv[2]
d = open(dll, 'rb').read()
os.makedirs(out, exist_ok=True)
mods = []; pos = 0
while True:
    i = d.find(b'\x03\x02\x23\x07', pos)
    if i < 0: break
    pos = i + 4
    if i % 4: continue
    ver, gen, bound, zero = struct.unpack_from('<4I', d, i + 4)
    if zero != 0 or ver >> 16 not in (1,) or bound > 1 << 22: continue
    p = i + 20; ok = True
    while p + 4 <= len(d):
        w = struct.unpack_from('<I', d, p)[0]
        op, wc = w & 0xffff, w >> 16
        if wc == 0 or op > 7000: break
        p += wc * 4
        if op == 56:   # OpFunctionEnd: the module ends unless another OpFunction follows
            if p + 4 > len(d) or struct.unpack_from('<I', d, p)[0] != (5 << 16 | 54): break
    blob = d[i:p]
    h = hashlib.sha256(blob).hexdigest()[:16]
    mods.append({'offset': i, 'size': len(blob), 'sha': h, 'version': '%d.%d' % ((ver >> 16) & 255, (ver >> 8) & 255), 'bound': bound})
    open(os.path.join(out, '%03d_%08x.spv' % (len(mods) - 1, i)), 'wb').write(blob)
    pos = p
json.dump(mods, open(os.path.join(out, 'index.json'), 'w'), indent=1)
print(len(mods), 'modules', len({m['sha'] for m in mods}), 'unique', sum(m['size'] for m in mods), 'bytes')
