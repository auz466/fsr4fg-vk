import sys, re, struct, numpy as np
def load(log):
    frames = []; cam = None; cur = None
    for line in open(log):
        if line.startswith('CAMERA'):
            cam = {k: [float(x) for x in v.split(',')] for k, v in re.findall(r'(\w+)=([-0-9.e+,]+)', line)}
        if 'FFX_MLFI_COMPUTE_FWD_MV' in line: cur = 'fwd'
        elif 'FFX_MLFI_DILATE' in line: cur = 'dil'
        elif line.startswith('LABEL begin'): cur = None
        m = re.search(r'ubo buf\d+ off=\d+ range=(\d+) data=(\S+)', line)
        if m and cur == 'fwd':
            d = bytes.fromhex(m.group(2))[:144]; f = struct.unpack('<32f', d[:128])
            frames.append({'cam': cam, 'A': np.array(f[:16]).reshape(4, 4), 'B': np.array(f[16:]).reshape(4, 4)}); cur = None
        if m and cur == 'dil':
            dil = struct.unpack('<2f', bytes.fromhex(m.group(2))[:8]); cur = None
            if cam is not None: cam['dil_jc'] = dil
    return frames
if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    for i, f in enumerate(load(sys.argv[1])):
        print('frame', i, {k: v for k, v in f['cam'].items() if k in ('pos', 'fwd', 'jitter', 'dil_jc')}); print(f['A']); print(f['B'])
