"""Builds the native library's assets from AMD's frame generation DLL (FSR SDK 2.4.0, FSR Frame Generation 4.0.2):
its SPIR-V shaders, rewritten where Mesa needs it (spv_fix.py), and its model weights. The DLL is read as data only.

    python3 -I tools/setup.py --dll /path/to/amd_fidelityfx_framegeneration_vk.dll [--out build/gen]

Writes <out>/assets.bin (every blob, end to end) and <out>/assets_gen.h (the tables that index it), and <out>/patched/:
every module of the DLL that Mesa needs rewritten, rewritten, named by the FNV-1a of AMD's original (the harness's
TRACE_PATCHES, so AMD's DLL runs under Wine on RADV).
"""
import argparse, hashlib, json, os, struct, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spv import Module, fnv1a64
import spv_fix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = json.load(open(os.path.join(ROOT, 'data', 'model.json')))
# (file offsets in the pinned DLL; the dilate pass has three variants, which AMD's DLL picks by the create flags:
# dilate_display with ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS, else dilate_jitter with
# ENABLE_MOTION_VECTORS_JITTER_CANCELLATION, else dilate)
# The watermark pass draws AMD's "MLFI DEBUG VIEW UNSUPPORTED" over the output under FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW.
PREP = {'dilate': 0x021b8b00, 'dilate_display': 0x021bbbc0, 'dilate_jitter': 0x021bf0f0, 'fwdmv': 0x021bcca0, 'splat': 0x021be070,
        'watermark': 0x021c14a0}

# what a shader's variable is bound to, by its name in AMD's SPIR-V
RESOURCES = ['SCRATCH', 'WEIGHTS', 'UBO_TENSOR', 'UBO_FI', 'SAMPLER', 'COLOR_PREV', 'COLOR', 'FLOW0', 'FLOW1', 'SPLAT0', 'SPLAT1',
             'HUDLESS', 'OUTPUT', 'MOTION', 'DEPTH', 'DISTORTION', 'DILATED_MV', 'DILATED_DEPTH', 'UBO_PREP', 'SPLAT_FLOW',
             'SPLAT_DEPTH', 'SPLAT_OUT', 'PREV_MV', 'PREV_DEPTH', 'SPLAT_1TO0', 'FWD_MV']
NAMES = {
    'ScratchBuffer': 'SCRATCH', 'InitializerBuffer': 'WEIGHTS', 'CsTensorSizes': 'UBO_TENSOR', 'CsFrameInterpolationConstants': 'UBO_FI',
    'g_sampler_linear': 'SAMPLER', 'g_sampler_nearest': 'SAMPLER', 'r_input_rgb0': 'COLOR_PREV', 'r_input_rgb1': 'COLOR',
    'rw_output_flow_0to05': 'FLOW0', 'rw_output_flow_1to05': 'FLOW1', 'r_input_packed_splatted0to05': 'SPLAT0',
    'r_input_packed_splatted1to05': 'SPLAT1', 'r_hudless_ui': 'HUDLESS', 'rw_prediction': 'OUTPUT',
    # the three preparation shaders
    'r_input_mv': 'MOTION', 'r_input_depth': 'DEPTH', 'r_distortion_field': 'DISTORTION', 'rw_output_dilated_mv': 'DILATED_MV',
    'rw_output_downscaled_depth': 'DILATED_DEPTH', 'CsDilate': 'UBO_PREP', 'CsAMDSplatting64Constants': 'UBO_PREP',
    'CsComputeFwdMVUsingMatConstants': 'UBO_PREP', 'r_input_flow': 'SPLAT_FLOW', 'r_depth': 'SPLAT_DEPTH', 'rw_splatted': 'SPLAT_OUT',
    'WatermarkCB': 'UBO_PREP', 'rw_target': 'OUTPUT', 'r_input_mv0': 'PREV_MV', 'r_input_depth0': 'PREV_DEPTH', 'r_packed_splatted1to0': 'SPLAT_1TO0', 'rw_output_mv_0_to_1': 'FWD_MV',
}
KINDS = ['SAMPLER', 'SAMPLED', 'STORAGE', 'UBO', 'SSBO']


def string(ws):
    b = struct.pack('<%dI' % len(ws), *ws)
    return b[:b.index(b'\0')].decode()


def carve(d):
    """Every SPIR-V module in the DLL: (offset, blob)"""
    pos = 0
    while True:
        i = d.find(b'\x03\x02\x23\x07', pos)
        if i < 0: return
        pos = i + 4
        if i % 4: continue
        ver, gen, bound, zero = struct.unpack_from('<4I', d, i + 4)
        if zero != 0 or ver >> 16 != 1 or bound > 1 << 22: continue
        p = i + 20
        while p + 4 <= len(d):
            w = struct.unpack_from('<I', d, p)[0]
            op, wc = w & 0xffff, w >> 16
            if wc == 0 or op > 7000: break
            p += wc * 4
            if op == 56 and (p + 4 > len(d) or struct.unpack_from('<I', d, p)[0] != (5 << 16 | 54)): break
        yield i, d[i:p]
        pos = p


def reflect(blob):
    """(entry point, [(binding, kind, resource)]) of a module, bindings in order"""
    m = Module(blob)
    names, binding, types, entry, empty = {}, {}, {}, None, False
    for x in m.ins:
        if x.op == 5: names[x.w[0]] = string(x.w[1:])
        elif x.op == 15: entry = string(x.w[2:])
        elif x.op == 71 and x.w[1] == 33: binding[x.w[0]] = x.w[2]
        elif x.op == 71 and x.w[1] == 34 and x.w[2] != 0: raise ValueError('a descriptor set other than 0')
        elif x.op in (25, 26, 30, 32): types[x.w[0]] = x
    binds = []
    for x in m.ins:
        if x.op != 59 or x.w[1] not in binding: continue
        ptr = types[x.w[0]]; pointee = types[ptr.w[2]]
        if pointee.op == 26: kind = 'SAMPLER'
        elif pointee.op == 25: kind = 'STORAGE' if pointee.w[6] == 2 else 'SAMPLED'
        else: kind = 'UBO' if x.w[2] == 2 else 'SSBO'
        name = names[x.w[1]]
        if name not in NAMES: raise ValueError('unknown shader variable ' + name)
        binds.append((binding[x.w[1]], kind, NAMES[name]))
    return entry, sorted(binds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dll', required=True)
    ap.add_argument('--out', default=os.path.join(ROOT, 'build', 'gen'))
    a = ap.parse_args()
    d = open(a.dll, 'rb').read()
    if hashlib.sha256(d).hexdigest() != MODEL['dll_sha256']:
        raise SystemExit('not the DLL this was made for: amd_fidelityfx_framegeneration_vk.dll of FSR SDK 2.4.0 (SHA-256 %s)' % MODEL['dll_sha256'])
    w = MODEL['weights']
    weights = d[w['offset']:w['offset'] + w['size']]
    if hashlib.sha256(weights).hexdigest() != w['sha256']: raise SystemExit('the weights are not where they were')
    by_entry, by_offset = {}, {}
    for off, blob in carve(d):
        by_offset[off] = blob
        m = Module(blob)
        e = next((string(x.w[2:]) for x in m.ins if x.op == 15), None)
        if e and e.startswith('mlfi_'): by_entry[e] = blob
    os.makedirs(a.out, exist_ok=True)
    patched = os.path.join(a.out, 'patched')
    os.makedirs(patched, exist_ok=True)
    for f in os.listdir(patched): os.remove(os.path.join(patched, f))
    for blob in by_offset.values():
        fixed = spv_fix.fix(blob)
        if fixed != blob: open(os.path.join(patched, '%016x.spv' % fnv1a64(blob)), 'wb').write(fixed)
    pack = bytearray()

    def add(blob):
        while len(pack) % 16: pack.append(0)
        off = len(pack); pack.extend(blob)
        return off, len(blob)

    def shader(blob, what):
        fixed = spv_fix.for_port(blob)
        tmp = os.path.join(a.out, 'check.spv')
        open(tmp, 'wb').write(fixed)
        if subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', tmp], capture_output=True).returncode != 0 and fixed != blob:
            raise SystemExit('a rewritten shader is not valid SPIR-V: ' + what)
        entry, binds = reflect(fixed)
        return add(fixed), entry, binds, fixed != blob

    out = ['// Generated by tools/setup.py from AMD\'s amd_fidelityfx_framegeneration_vk.dll (FSR SDK 2.4.0). Not edited by hand.', '#pragma once', '']
    out.append('enum Res : uint8_t { %s, RES_COUNT };' % ', '.join('RES_' + r for r in RESOURCES))
    out.append('enum Kind : uint8_t { %s };' % ', '.join('KIND_' + k for k in KINDS))
    out.append('struct GenBind { uint16_t binding; Kind kind; Res res; };')
    out.append('struct GenTerm { char kind; uint32_t a, b, c, d, e; };   // k: a | t: ceil(T[a][b] / c) * d | p: the padding count over T[a], max (b, c), clamp (d, e)')
    out.append('struct GenPass { uint32_t offset, size; const char* entry; GenTerm groups[3]; uint32_t bind_count; GenBind binds[12]; };')
    out.append('')
    rewritten = 0

    def emit(name, off_size, entry, binds, terms):
        t = ', '.join("{'%s', %s}" % (x[0][0], ', '.join(str(v) for v in (list(x[1:]) + [0] * 5)[:5])) for x in terms)
        b = ', '.join('{%d, KIND_%s, RES_%s}' % x for x in binds)
        return '    {%d, %d, "%s", {%s}, %d, {%s}},' % (off_size[0], off_size[1], entry, t, len(binds), b)

    for variant in ('4k', '8k'):
        rows = []
        for p in MODEL['passes'][variant]:
            blob = by_entry[p['entry']]
            if len(Module(blob).ins[Module(blob).first_function():]) <= 4 and not reflect(blob)[1]:
                continue   # (an empty pass: nothing bound, nothing done)
            os_, entry, binds, changed = shader(blob, p['entry'])
            rewritten += changed
            rows.append(emit(p['entry'], os_, entry, binds, p['groups']))
        out.append('static const GenPass GEN_PASSES_%s[] = {' % variant.upper()); out += rows; out.append('};'); out.append('')
        print(variant, len(rows), 'passes')
    for name, off in PREP.items():
        os_, entry, binds, changed = shader(by_offset[off], name)
        rewritten += changed
        out.append('static const GenPass GEN_%s = ' % name.upper() + emit(name, os_, entry, binds, [['k', 1]] * 3).strip().rstrip(',') + ';')
    wo = add(weights)
    out.append('')
    out.append('static const uint32_t GEN_WEIGHTS_OFFSET = %d, GEN_WEIGHTS_SIZE = %d;' % wo)
    out.append('static const uint64_t GEN_SCRATCH_BYTES_4K = %dull, GEN_SCRATCH_BYTES_8K = %dull;' % (MODEL['scratch_bytes']['4k'], MODEL['scratch_bytes']['8k']))
    out.append('// how each of the 92 tensor sizes follows the display size: see tensor_size() in fsr4fg.cpp')
    codes = ['A0', 'A1', 'A2', 'A3', 'A4', 'F', 'G', 'G2', 'Gh', 'D']
    out.append('enum TensorRule : uint8_t { %s };' % ', '.join('T_' + c for c in codes))
    out.append('static const TensorRule GEN_TENSOR_RULES[92] = { %s };' % ', '.join('T_' + c for c in MODEL['tensor_sizes']))
    open(os.path.join(a.out, 'assets_gen.h'), 'w').write('\n'.join(out) + '\n')
    open(os.path.join(a.out, 'assets.bin'), 'wb').write(pack)
    os.remove(os.path.join(a.out, 'check.spv'))
    print('assets: %d bytes, %d shaders rewritten for Mesa; %d modules in %s for the harness' % (len(pack), rewritten, len(os.listdir(patched)), patched))


if __name__ == '__main__':
    main()
