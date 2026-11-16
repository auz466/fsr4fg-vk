# Third-party notices

This file is started in workstream 3.2 and completed in 3.7 (the embedded shaders and weights, the frame generation
swapchain).

## AMD FidelityFX SDK 2.4.0 (MIT)

- `amd_fidelityfx_framegeneration_vk.dll`: `tools/setup.py` takes its shaders and model weights at build time and the
  `.so` embeds them. AMD's `docs/license.md` lists this DLL under its MIT section. A built `.so` must be accompanied by
  this notice.
- `third_party/amd-fidelityfx-api/`: AMD's FidelityFX API headers, unmodified. Their notice is beside them.
- `src/fsr4fg.cpp`, `fsr3_memory_usage_v2`: adapted from AMD's FidelityFX SDK 2.4.0,
  `Kits/FidelityFX/framegeneration/fsr3/internal/ffx_frameinterpolation.cpp` (`ffxFrameInterpolationGetGpuMemoryUsage`,
  its resource descriptions), `ffx_opticalflow.cpp` (`ffxOpticalflowGetGpuMemoryUsage`, its resource descriptions)
  and `Kits/FidelityFX/backend/vk/ffx_vk.cpp` (`ffxGetResourceSizeFromDescriptionVK`), so that the GPU memory usage
  query (V2) answers as AMD's DLL does.

AMD's licence for these files (the MIT section of the SDK's `docs/license.md`):

Copyright (C)  Advanced Micro Devices, Inc. All rights reserved. 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: 

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. 

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 