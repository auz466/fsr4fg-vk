"""Rewrites for AMD's SPIR-V so Mesa takes it.

half_images: an image whose sampled type is a 16-bit float is not valid Vulkan SPIR-V (the sampled type must be a 32-bit
float) and RADV's compiler faults on it. The image becomes a 32-bit float one, and each read from or write to it converts.
"""
import os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spv import Module, Ins, fnv1a64

IMAGE_READS = (87, 88, 95, 96, 98)   # ImageSampleImplicitLod, ExplicitLod, Fetch, Gather, Read


def half_images(m):
    floats = {x.w[1]: x.w[0] for x in m.ins if x.op == 22}
    half = floats.get(16)
    if half is None or not any(x.op == 25 and x.w[1] == half for x in m.ins): return False
    f32 = floats.get(32)
    first_image = next(i for i, x in enumerate(m.ins) if x.op == 25 and x.w[1] == half)
    if f32 is None:
        f32 = m.new_id()
        m.ins.insert(first_image, Ins(22, [f32, 32]))
    elif next(i for i, x in enumerate(m.ins) if x.op == 22 and x.w[0] == f32) > first_image:
        raise ValueError('the 32-bit float is declared after the image that needs it')
    vectors = {(x.w[1], x.w[2]): x.w[0] for x in m.ins if x.op == 23}
    half_types = {half: 1}
    half_types.update({tid: n for (comp, n), tid in vectors.items() if comp == half})
    added = []

    def as_float(t):
        n = half_types[t]
        if n == 1: return f32
        if (f32, n) not in vectors:
            vectors[(f32, n)] = m.new_id()
            added.append(Ins(23, [vectors[(f32, n)], f32, n]))
        return vectors[(f32, n)]

    value_type = {}
    for x in m.ins:
        rt, rid, _ = x.layout()
        if rt is not None and rid is not None: value_type[x.w[rid]] = x.w[rt]
    out = []
    for x in m.ins:
        if x.op == 25 and x.w[1] == half:
            x.w[1] = f32
        elif x.op in IMAGE_READS and x.w[0] in half_types:
            t, rid = x.w[0], x.w[1]
            tmp = m.new_id()
            x.w[0], x.w[1] = as_float(t), tmp
            out += [x, Ins(115, [t, rid, tmp])]   # OpFConvert back to the half the rest of the shader expects
            continue
        elif x.op == 99 and value_type.get(x.w[2]) in half_types:
            tmp = m.new_id()
            out.append(Ins(115, [as_float(value_type[x.w[2]]), tmp, x.w[2]]))
            x.w[2] = tmp
        out.append(x)
    m.ins = out
    f = m.first_function()
    m.ins[f:f] = added
    m.dedupe_types()
    return True


def sampled_int64(m):
    """A sampled image of 64-bit integers: Mesa has no 64-bit texel fetch (RADV's compiler faults). The image is read as
    two 32-bit halves instead (the caller binds an R32G32_UINT view of the R64_UINT image) and put back together."""
    ints = {(x.w[1], x.w[2]): x.w[0] for x in m.ins if x.op == 21}
    targets = [x for x in m.ins if x.op == 25 and x.w[1] in (ints.get((64, 0)), ints.get((64, 1))) and x.w[6] == 1]
    if not targets: return False
    u64 = ints[(64, 0)]
    if any(x.w[1] != u64 for x in targets): raise ValueError('a signed 64-bit sampled image')
    types = []   # declarations to add before the first function
    u32 = ints.get((32, 0))
    if u32 is None: raise ValueError('no 32-bit uint')
    vectors = {(x.w[1], x.w[2]): x.w[0] for x in m.ins if x.op == 23}
    v4u32 = vectors.get((u32, 4))
    if v4u32 is None:
        v4u32 = m.new_id(); types.append(Ins(23, [v4u32, u32, 4]))
    v4u64 = vectors.get((u64, 4))
    c32 = next((x.w[1] for x in m.ins if x.op == 43 and x.w[0] == u64 and x.w[2:] == [32, 0]), None)
    if c32 is None:
        c32 = m.new_id(); types.append(Ins(43, [u64, c32, 32, 0]))
    first = min(i for i, x in enumerate(m.ins) if x in targets)
    if next(i for i, x in enumerate(m.ins) if x.op == 21 and x.w[0] == u32) > first:
        raise ValueError('the 32-bit uint is declared after the image that needs it')
    for x in targets: x.w[1] = u32
    out = []
    for x in m.ins:
        if x.op == 95 and x.w[0] == v4u64:
            rid = x.w[1]
            tmp, lo, hi, lo64, hi64, sh, both = (m.new_id() for _ in range(7))
            x.w[0], x.w[1] = v4u32, tmp
            out += [x, Ins(81, [u32, lo, tmp, 0]), Ins(81, [u32, hi, tmp, 1]), Ins(113, [u64, lo64, lo]), Ins(113, [u64, hi64, hi]),
                    Ins(196, [u64, sh, hi64, c32]), Ins(197, [u64, both, lo64, sh]), Ins(80, [v4u64, rid, both, both, both, both])]
            continue
        if x.op in (87, 88, 96, 98) and x.w[0] == v4u64 and False: raise ValueError('unhandled 64-bit read')
        out.append(x)
    m.ins = out
    f = m.first_function()
    m.ins[f:f] = types
    m.dedupe_types()
    return True


def output_without_format(m):
    """The last pass writes the caller's output (rw_prediction) through an image declared Rgba32f, whatever the
    caller's format: undefined in Vulkan when the view's format differs (RADV converts as if it matched). The image is
    declared with no format instead, which writes through the view's own (StorageImageWriteWithoutFormat).
    Only for the port's own copy of the shaders: AMD's DLL runs under Wine with its shaders as they are."""
    names = {x.w[0]: string_of(x.w[1:]) for x in m.ins if x.op == 5}
    var = next((x for x in m.ins if x.op == 59 and names.get(x.w[1]) == 'rw_prediction'), None)
    if var is None: return False
    ptr = next(x for x in m.ins if x.op == 32 and x.w[0] == var.w[0])
    image = next(x for x in m.ins if x.op == 25 and x.w[0] == ptr.w[2])
    if image.w[7] == 0: return False
    if sum(1 for x in m.ins if x.op == 32 and x.w[2] == image.w[0]) != 1 or sum(1 for x in m.ins if x.op == 59 and x.w[0] == ptr.w[0]) != 1:
        raise ValueError("rw_prediction's image type is shared")
    loads = {x.w[1] for x in m.ins if x.op == 61 and x.w[2] == var.w[1]}
    if any(x.op == 98 and x.w[2] in loads for x in m.ins): raise ValueError('rw_prediction is read')
    image.w[7] = 0   # Unknown
    if not any(x.op == 17 and x.w[0] == 56 for x in m.ins):
        m.ins.insert(next(i for i, x in enumerate(m.ins) if x.op == 17), Ins(17, [56]))   # StorageImageWriteWithoutFormat
    return True


def string_of(ws):
    b = struct.pack('<%dI' % len(ws), *ws)
    return b[:b.index(b'\0')].decode()


def for_port(blob):
    """The port's copy of a shader: fix(), and what the spec needs beyond it (output_without_format)."""
    m = Module(fix(blob))
    return m.blob() if output_without_format(m) else fix(blob)


def fix(blob):
    m = Module(blob)
    changed = half_images(m)
    changed = sampled_int64(m) or changed
    return m.blob() if changed else blob


if __name__ == '__main__':
    import glob, subprocess
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(src, '*.spv'))):
        blob = open(f, 'rb').read()
        if Module(blob).blob() != blob: raise SystemExit('round trip failed: ' + f)
        fixed = fix(blob)
        if fixed == blob: continue
        name = os.path.join(dst, '%016x.spv' % fnv1a64(blob))
        open(name, 'wb').write(fixed)
        r = subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', name], capture_output=True, text=True)
        print(os.path.basename(f), '->', os.path.basename(name), 'valid' if r.returncode == 0 else r.stderr.strip()[:300])
        n += 1
    print(n, 'modules rewritten')
