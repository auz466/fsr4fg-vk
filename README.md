# fsr4fg-vk

An independent, native Linux implementation of AMD FSR Frame Generation 4.0.2 (FidelityFX SDK 2.4.0) for Vulkan. It
exports AMD's FSR API entry points (`ffxCreateContext`, `ffxDestroyContext`, `ffxConfigure`, `ffxQuery`, `ffxDispatch`)
so a program that loads `amd_fidelityfx_framegeneration_vk` can load this instead. It is not made or endorsed by AMD.

AMD's shaders and model weights are not in this repository. You build the library from your own copy of AMD's DLL.

## Build your own `.so`

You need `amd_fidelityfx_framegeneration_vk.dll` from FidelityFX SDK 2.4.0 (SHA-256 `d002dabf…98b4`; the script checks
it), plus `cmake`, a C++17 compiler, `python3` and the Vulkan headers.

```sh
./make-so.sh /path/to/amd_fidelityfx_framegeneration_vk.dll          # -> dist/amd_fidelityfx_framegeneration_vk.so
./make-so.sh /path/to/amd_fidelityfx_framegeneration_vk.dll ~/out    # -> ~/out/amd_fidelityfx_framegeneration_vk.so
```

The DLL is only read as data, never run. The resulting `.so` embeds AMD's shaders and weights, so keep it for your own
use and do not redistribute it.

`build.sh` is the developer build (also builds the trace harness); see `STANDALONE_PLAN.md`.

## Notices

This repository's own code is MIT licensed (`LICENSE`). AMD's headers and the parts adapted from AMD's SDK keep AMD's
MIT notice (`THIRD_PARTY_NOTICES.md`). AMD's DLL and what is extracted from it are AMD's and are not covered.
