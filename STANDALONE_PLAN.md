# Plan: fsr4fg-vk as a standalone library

This is a brief for an agent that picks up `fsr4fg-vk` and turns it from a Hookshot-specific experiment into a library
that other Linux Vulkan programs can use. Read all of it before starting. Work through the workstreams in order. Each one
says what "done" means.

## 1. What exists today

`fsr4fg-vk` is a native Linux implementation of **AMD FSR Frame Generation 4.0.2**, the machine-learning frame
generation in AMD's FSR SDK 2.4.0. AMD ships it only as a Windows DLL (`amd_fidelityfx_framegeneration_vk.dll`).

- `src/fsr4fg.cpp` (about 1050 lines) builds `amd_fidelityfx_framegeneration_vk.so`. It exports AMD's five FSR API entry
  points (`ffxCreateContext`, `ffxDestroyContext`, `ffxConfigure`, `ffxQuery`, `ffxDispatch`). It records the same
  compute passes AMD's DLL records, over AMD's own SPIR-V shaders and model weights.
- `tools/setup.py` extracts those shaders and weights from AMD's DLL at build time. It reads the DLL as data only and
  never runs it. It checks the DLL's SHA-256 (`d002dabf…98b4`). It produces `build/gen/assets.bin` and
  `build/gen/assets_gen.h`, which are embedded in the `.so`.
- `tools/spv_fix.py` rewrites 7 of AMD's shaders that crash RADV's compiler: half-float image types, and a sampled 64-bit
  image that is read as two 32-bit halves instead.
- `data/model.json` holds everything read off AMD's DLL at run time: the rules for the 92 tensor sizes, each pass's
  dispatch size, where the weights are, and the scratch buffer sizes.
- `harness/trace.cpp` is the ground truth. It runs any FSR API frame generation provider on a synthetic scene, logs every
  device-level Vulkan call (pipelines, bindings, constant buffer bytes, dispatch sizes), and dumps images and buffers.
  It is built twice:
  - for Windows with mingw (`harness/trace.exe`), to run AMD's real DLL under Wine (`harness/run_amd.sh <name> W H RW RH frames dumpFrom`);
  - natively (`build/trace`), to run the port.
- `tools/compare.py <amd run> <port run>` compares the generated frames. `tools/at_pass.sh <n>` compares the scratch
  buffer after network pass `n` of the first frame, which is how divergences are bisected.
- `build.sh [path to AMD's DLL]` builds the assets (first time), the library and the native harness.

**Verified:** the port's generated frames are bit-identical to AMD's DLL under Wine (RX 9070 XT, RADV, Mesa 26.2.6) for:
a still camera at 1920x1080, a moving camera at 2048x1152 output from a 1364x768 render, and 5120x2880 (the 8k
network). The only differences ever seen came from the harness's scene, which `sin()` draws slightly differently in the
mingw and glibc builds (33 bytes on one frame).

**Integrated:** Hookshot (`/home/austin/Games/hookshot-src`, branch `local-fg`) loads it through `gpu/vk/vk_ffx.cpp`. It
has not been confirmed making frames in gameplay yet. That is the user's playtest and not part of this plan.

**Background:** `/home/austin/Games/HOOKSHOT_FSR4_HANDOFF.md` explains how the port was made and the pitfalls found.

### Facts that cost time to find

- The network's 191 passes (16 of them empty and skipped) must be built with `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`
  set to 32. On wave64 the results go to NaN from pass 26.
- AMD's Vulkan build gives every pass one sampler, nearest-filtered with clamp-to-edge addressing, whatever the shader
  calls it. The port does the same (`FSR4FG_LINEAR=1` overrides it).
- Every constant block layout comes from the debug names in AMD's SPIR-V (`tools/spv_structs.py`). The camera matrices
  were fitted bit for bit from traces (3.1); the earlier fit only held because the old harness scene never made the
  camera matter:
  - all single precision; products accumulated in index order; a cofactor (MESA-form) inverse scaled by `1/det`;
  - `M[n]` (clip to world) `= inverse(P) * [right; up; forward; position]` (rows are vectors);
  - `P` left-handed: `b = 1/tanf(fov/2)`, `a = b / (renderW/renderH)`, `P23 = 1`; finite planes `P22 = f/(f-n)`,
    `P32 = n*f/(n-f)` (n and f swapped when inverted); infinite `P22, P32 = 1, -near` or `0, near` (inverted);
  - `prevPsToPs = inverse(M[n] * inverse(M[n-1]))`, `prevPsToPrev2Ps = M[n-1] * inverse(M[n-2])`;
  - missing history is the identity; a reset does not clear it.
  - AMD's forward motion pass uses these only where the motion vectors agree with the camera within 2 pixels.
- The splat images are `R64_UINT` to their writer and `R32G32_UINT` to their readers. The port creates them as
  `R32G32_UINT`; the 64-bit atomics shader is never rewritten.
- AMD's DLL under Wine only works with the 7 rewritten shaders. The harness substitutes them by FNV-1a hash of the
  original module, with `TRACE_PATCHES=<dir of re/patched>`.

## 2. Ground rules

- **AMD's DLL is the specification.** Every behaviour you add must first be traced from AMD's DLL with the harness, with
  the same inputs, and the port must then match it. Bit-exact is the bar. Where exactness is impossible, explain why in
  the code. Do not guess constants from the SDK documentation alone.
- **Never commit or redistribute AMD's DLL, `build/gen/`, `re/` or anything extracted.**
  - `re/` is 3.8 GB of traces, kept locally only.
  - The built `.so` embeds AMD's shaders and weights. AMD's licence (`/home/austin/Games/fsr4-deps/amd-license-2.4.0.md`)
    lists the signed DLLs under its MIT section, so distributing a built binary needs AMD's MIT notice beside it.
    Confirm that reading before any release.
- **Ask the user before anything outward-facing:** creating a GitHub repository, pushing, publishing a release, or
  choosing a licence. The user owns these decisions. Keep the work local until they say otherwise.
- Don't brand it as AMD's. The file name stays `amd_fidelityfx_framegeneration_vk.so` so it drops in where programs look
  for AMD's, but the README, logs and the version name (`4.0.2 (fsr4fg-vk)`) say it is an independent implementation.
- Don't touch `/home/austin/Games/Hookshot` or `hookshot-src` except to re-test the integration at the end, and tell the
  user before you replace the installed `.so` there.
- AMD's DLL runs under Wine with `WINEPREFIX=/home/austin/Games/fsr4-deps/wineprefix`. AMD's SDK sources (API headers,
  the open FSR 3 frame generation and its Vulkan swapchain) are sparse-cloned at
  `/home/austin/Games/fsr4-deps/amd/sdk-src` (tag v2.4.0).

## 3. Workstreams

### 3.0 Make it a project

1. `git init` in `fsr4fg-vk`, with a `.gitignore` for `build/`, `re/`, `harness/*.dll`, `harness/*.exe` and `harness/trace2.exe`.
2. Move `include/ffx/` (AMD's MIT API headers) under `third_party/amd-fidelityfx-api/`, with AMD's licence text, and keep
   the include paths working.
3. Replace `build.sh` with CMake (keep `build.sh` as a thin wrapper if convenient). Targets:
   - the library;
   - the native harness;
   - an `assets` step that runs `tools/setup.py` when `assets.bin` is missing or the DLL path changes.
4. Remove hard-coded paths:
   - `/home/austin/...` in `harness/run_amd.sh`: use the environment or a config file;
   - the SPIR-V grammar path in `tools/spv.py`: find it, falling back to `/usr/share` or `/usr/include`.
5. Tidy the leftovers: the unused `UBO_ALIGN`; the tautological pointer compare in `harness/trace.cpp`'s feature-chain
   lambda (rewrite it).
6. Leave the `LICENSE` file out until the user chooses one, and say so in the report.

Done when a clean clone plus AMD's DLL builds with one command, and nothing ignored is tracked.

### 3.1 A regression suite against AMD

1. Make the harness scene deterministic across mingw and glibc. Use integer or table-driven patterns, or generate it in
   Python once and load it from a file, so input pictures are byte-identical in both builds.
2. Add `tests/against_amd.py`. It runs a matrix of cases through AMD's DLL under Wine and through the port, caches
   AMD's results under `re/` keyed by case, and fails on any byte difference in the generated frames. Cases:
   - sizes: 1280x720, 1920x1080, 2560x1440, 3840x2160, 5120x2880, and an odd one (1001x777);
   - render scale: native, and about 0.67;
   - camera: still, and moving and turning;
   - a reset in the middle of the stream;
   - the four depth modes: inverted or not, infinite or not.

   The harness already supports these through `TRACE_*` environment variables; see the top of its `main()` and
   `harness/sweep.sh`.
3. Add a Vulkan validation-layer run of the port: `VK_LAYER_KHRONOS_validation`, including synchronisation validation.
   Install `vulkan-validation-layers` if it's missing, after checking with the user. Fix whatever it reports.

Done when the suite passes at every case and the validation run is clean.

### 3.2 Finish the inputs the API defines

Trace each feature from AMD first, then implement it, then add it to the suite.

- **Create flags.** `FFX_FRAMEGENERATION_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION` probably fills the dilate pass's
  `motionVectorJitterCancellation`; `ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS` probably changes its render size and
  scale. Today both only print a warning, so callers silently get wrong frames. That is the worst current gap.
- **Colour.** The PQ and scRGB transfer functions with `minMaxLuminance`, the `ENABLE_HIGH_DYNAMIC_RANGE` flag, and the
  back-buffer formats `A2B10G10R10_UNORM`, `R16G16B16A16_SFLOAT` and `B8G8R8A8_UNORM/SRGB`.
- **A HUD-less colour.** `ffxConfigureDescFrameGeneration.HUDLessColor` and
  `ffxCreateContextDescFrameGenerationHudless`. The last network pass binds `r_hudless_ui`, and the constants have
  `hudlessAttachedFactor`.
- **A distortion field.** `ffxConfigureDescFrameGenerationRegisterDistortionFieldResource`; the dilate shader binds
  `r_distortion_field`.
- **A partial `generationRect`** (letterboxed output), and frame IDs that skip or repeat. The API says a gap of more than
  one resets; check what AMD actually does.
- **Debug flags.** Tear lines, reset indicators and pacing lines, as AMD draws them. The debug view itself is
  unsupported in 4.0.x per AMD's docs.
- **More than one generated frame.** Trace AMD with `TRACE_GEN=2..4`. The network's flow outputs are named `0to05` and
  `1to05`, which suggests a fixed midpoint. Find out whether AMD makes several frames, refuses, or repeats one, and do
  the same.
- **Version queries.** The GPU memory usage queries (V1 and V2) should return what AMD's do. Compare the numbers from
  the trace.

### 3.3 Hardware checks and robustness

- In `ffxQuery(GET_VERSIONS)` and `ffxCreateContext`, check the device. It needs:
  - `VK_KHR_cooperative_matrix` with FP8 cooperative matrices (`VK_EXT_shader_float8`);
  - `shaderImageInt64Atomics`;
  - subgroup size control that allows 32;
  - 16-bit and 8-bit types;
  - the Vulkan memory model.

  With anything missing, return zero providers and a clear message, not a crash at pipeline creation. Note that the
  caller must also have *enabled* these features on its device; document that, and check what it can (a pipeline
  creation failure must turn into an error code).
- Lift the hard-coded ring of 8 descriptor/constant slots: size it from the frames in flight, or derive slot reuse from
  `frameID`. Document the assumption either way.
- Make concurrent use of separate contexts safe. One context stays single-threaded, as AMD's is; document that.
- Handle allocation failures and `VK_ERROR_DEVICE_LOST` without leaking. Destroying a context must not need a device
  idle that the caller did not ask for.
- Use the caller's `ffxAllocationCallbacks` for host allocations.

### 3.4 Performance

- Measure GPU time per frame (timestamps around prepare and generate) against AMD's DLL under Wine at 1080p, 1440p and
  4K. AMD's documentation quotes 2.2 ms at 4K on an RX 9070 XT.
- The port issues a full memory barrier before every pass (`run()` in `fsr4fg.cpp`). AMD's trace shows which resources
  each pass reads and writes. Replace the blanket barriers with the dependencies that exist, and keep the suite
  bit-exact.
- Pipeline creation for 175 passes on first use: measure it, and add an optional `VkPipelineCache` saved under
  `$XDG_CACHE_HOME`.

### 3.5 The frame generation swapchain (the large piece)

Most programs don't call frame generation directly. They use AMD's replacement swapchain, which presents and paces the
real and generated frames. It is created with `ffxCreateContextDescFrameGenerationSwapChainVK`, and its functions are
replaced through `FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK` (see
`framegeneration/include/vk/ffx_api_framegeneration_vk.h`).

- AMD's FSR 3 Vulkan swapchain is open source under MIT, in `sdk-src/Kits/FidelityFX/framegeneration/fsr3/vk/`
  (`FrameInterpolationSwapchainVK.cpp` and the files beside it). Adapt it, keeping AMD's notice, rather than writing one
  from scratch. Its frame generation callback is the seam where the 4.0.2 context plugs in, set through
  `ffxConfigureDescFrameGeneration.frameGenerationCallback`.
- AMD's own 4.0 swapchain ("driver-assisted frame pacing") needs AMD's Windows driver extensions
  (`vkSetFramePacingConfigAMD` and similar), which RADV does not have. Use the FSR 3.1 swapchain's CPU pacing, and say so.
- This needs a windowed test program: a small SDL or XCB Vulkan app that renders the harness scene, presents through the
  replacement swapchain, and logs present timestamps (`VK_KHR_present_wait` if available). There is no Linux sample from
  AMD to compare against. Test on both X11 and Wayland.
- Ship it as a separate library (`amd_fidelityfx_framegeneration_swapchain_vk.so`), or in the same one, keyed by
  description type. Decide based on how programs load it, and note the choice.

### 3.6 Optional: a loader

Programs that link AMD's `amd_fidelityfx_loader_vk` call one library that routes by effect. A small
`amd_fidelityfx_loader_vk.so` shim could do that routing:

- upscaling to fsr4-vk's `amd_fidelityfx_upscaler_vk.so`, if present;
- frame generation to this library.

The shim would merge `GET_VERSIONS` answers. Do this only if 3.0 to 3.5 are done.

### 3.7 Documentation and release prep

- A `README.md` covering:
  - what it is;
  - that it is independent and not AMD's;
  - requirements (GPU, Mesa version, device features to enable);
  - how to build (where to get AMD's SDK 2.4.0 DLL);
  - the supported API surface and its limits;
  - environment variables (`FSR4FG_LOG`, `FSR4FG_VERBOSE`, `FSR4FG_LINEAR`).
- `THIRD_PARTY_NOTICES.md` with AMD's MIT notice for the headers, for anything adapted from the FSR 3 swapchain, and for
  the embedded shaders and weights.
- A short `HOW_IT_WAS_MADE.md` explaining the tracing method, so the next AMD release can be redone. The same harness
  traces a new DLL. `data/model.json` and the DLL hash change, and the fitting scripts (`tools/fit_groups.py`,
  `tools/sweep_table.py`, `tools/cam_table.py`) regenerate them. Script that regeneration end to end.
- Then stop and hand the user the release questions: licence, repository name and host, and whether to publish.

## 4. Reporting

At the end of each workstream, report:

- what changed;
- the suite result (cases passed, bit-exact or not);
- anything AMD's DLL did that surprised you;
- anything you could not verify.

Don't describe a feature as done if it was only compiled. "Done" means traced from AMD and matched.
