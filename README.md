# fsr4fg-vk

A native Linux Vulkan implementation of **AMD FSR Frame Generation 4.0.2** (the machine-learning one, FidelityFX SDK
2.4.0). AMD ships it for Vulkan only as a Windows DLL (`amd_fidelityfx_framegeneration_vk.dll`); this library answers the
same FSR API calls with AMD's own shaders and model weights, which the build takes out of that DLL.

This is an independent implementation. It is not AMD's, and AMD does not support it. It reports itself as
`4.0.2 (fsr4fg-vk)`.

It drops in where a program looks for AMD's `amd_fidelityfx_framegeneration_vk`: the same file name
(`amd_fidelityfx_framegeneration_vk.so`) and the same entry points (`ffxCreateContext`, `ffxDestroyContext`,
`ffxConfigure`, `ffxQuery`, `ffxDispatch`). Its sibling for upscaling is [fsr4up-vk](https://github.com/auz466/fsr4up-vk).

Heavily assisted by Claude.

## Status

- Generated frames **bit-identical to AMD's DLL** (run under Wine on the same GPU) on the regression suite
  (`tests/against_amd.py`): sizes from 1001x777 to 5120x2880 (the 4k and 8k networks), native and scaled render sizes,
  still and moving camera, resets, the four depth modes, jitter cancellation, display-sized motion vectors and the other
  create and dispatch flags, plus every API answer.
- Tested on an AMD Radeon RX 9070 XT (RDNA 4) with RADV (Mesa 26.2.6). The only differences ever seen came from the
  harness's synthetic scene, which `sin()` draws slightly differently in the mingw and glibc builds.
- Not confirmed in gameplay in every program that can load it; Hookshot is the first one it was integrated into.

## Requirements

A GPU and driver with Vulkan 1.3, `VK_KHR_cooperative_matrix`, `VK_EXT_shader_float8` and
`VK_EXT_shader_image_atomic_int64`, and 32-wide compute subgroups (subgroup size control). The caller must have
**enabled** these on its `VkDevice`: `cooperativeMatrix`, `shaderFloat8`, `shaderFloat8CooperativeMatrix`,
`shaderImageInt64Atomics`, `shaderInt64`, `shaderInt8`, `shaderFloat16`, `vulkanMemoryModel`, `variablePointers`,
`variablePointersStorageBuffer`, `shaderStorageImageWriteWithoutFormat`, `subgroupSizeControl`. A device that cannot run
it gets no provider: GET_VERSIONS lists none and create returns `FFX_API_RETURN_NO_PROVIDER`, as AMD's DLL does.

## Build

This repository holds no AMD data. The build makes the `.so` on your machine, from your own copy of AMD's DLL.

Needs CMake, Python 3, a C++20 compiler (clang++ or g++), the Vulkan headers and `curl`
(Arch: `cmake python vulkan-headers`; Debian/Ubuntu: `cmake python3 libvulkan-dev g++`).

```sh
git clone https://github.com/auz466/fsr4fg-vk && cd fsr4fg-vk
./make-so.sh
```

Output: `dist/amd_fidelityfx_framegeneration_vk.so`. The script downloads
`Kits/FidelityFX/signedbin/amd_fidelityfx_framegeneration_vk.dll` (37 MB) from AMD's
[FidelityFX SDK v2.4.0](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/tree/v2.4.0/Kits/FidelityFX/signedbin)
into `build-so/dll/`, checks its SHA-256 (`d002dabff8c5e71eaa715fc99438cd4dc81230262a7897b16bc402efec3498b4`), reads it as
data (never runs it), and embeds AMD's shaders and weights in the library. Later runs reuse it.

Options: `./make-so.sh /path/to/amd_fidelityfx_framegeneration_vk.dll` uses your own copy, `--download` refetches, and a
second argument is the output directory. The script also copies `THIRD_PARTY_NOTICES.md` (AMD's MIT notice) beside the `.so`.
`./build.sh [dll]` is the developer build, which also builds the test harness (`build/trace`, and
`build/wine/trace.exe` with mingw-w64).

## Environment variables

- `FSR4FG_LOG=<file>`: append the library's messages to a file.
- `FSR4FG_VERBOSE=1`: print them to stderr.
- `FSR4FG_PIPELINE_CACHE`: where the pipeline cache goes (default `$XDG_CACHE_HOME/fsr4fg-vk`, else `~/.cache/fsr4fg-vk`;
  anything else is that directory).
- `FSR4FG_LINEAR=1`: linear filtering in the passes' sampler instead of AMD's nearest. For testing.
- `FSR4FG_ALL_BARRIERS=1`: a barrier after every pass. For testing.

## Testing

- `tests/against_amd.py`: the regression suite against AMD's DLL under Wine (`harness/run_amd.sh`; set
  `FSR4FG_WINEPREFIX` in `local.env`). Needs `build/wine/trace.exe` (mingw-w64).
- `tests/validation.sh`: the port under the Khronos validation layer with synchronisation validation.
- `tests/perf.py`: GPU time per frame, the port against AMD's DLL.

See `STANDALONE_PLAN.md` for how it was made and its pitfalls, and `THIRD_PARTY_NOTICES.md` for AMD's notice.

## Licence

MIT, see `LICENSE` (this repository's own code). AMD's headers in `third_party/` and the shaders and weights a build
takes from AMD's DLL are AMD's, under AMD's MIT licence: the FidelityFX SDK's
[`docs/license.md`](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/main/docs/license.md) lists
`amd_fidelityfx_framegeneration_vk.dll` among the files under it. A built `.so` embeds them, so anyone who passes one on
must keep AMD's notice with it (`THIRD_PARTY_NOTICES.md`). This repository itself ships no binary.
