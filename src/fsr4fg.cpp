// fsr4fg-vk: a native Vulkan provider of AMD FSR Frame Generation 4.0.2 (the machine-learning one), for Linux.
//
// AMD ships this effect only as a Windows DLL (amd_fidelityfx_framegeneration_vk.dll, FSR SDK 2.4.0). This library answers
// the same FSR API entry points (ffxCreateContext, ffxConfigure, ffxQuery, ffxDispatch, ffxDestroyContext) and records the
// same compute passes over the same shaders and model weights, which tools/setup.py takes out of AMD's DLL at build time.
// What the passes are, in what order, over which images, with which constants, was read off AMD's DLL at work (harness/
// trace.cpp): nothing here is AMD's code.
//
// A frame is two dispatches, as AMD's documentation orders them:
//   prepare  (ffxDispatchDescFrameGenerationPrepareV2): the game's motion and depth dilated to a third of the display
//            size, splatted forward, and the previous frame's motion carried to this one through the cameras.
//   generate (ffxDispatchDescFrameGeneration): the network, 175 passes over one scratch buffer. It reads the previous and
//            the current picture, estimates the flow to the midpoint, and writes the picture in between.
//
// Not done: AMD's swapchain and pacing (the caller presents), more than one made frame an interval, HUD-less colour,
// distortion fields, the debug overlays other than the tear lines' colour.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include <dlfcn.h>

// (AMD's header marks the entry points __declspec(dllexport): here that is default visibility, with everything else hidden)
#define __declspec(x) __attribute__((visibility("default")))
#include "ffx/api/include/ffx_api.h"
#include "ffx/api/include/ffx_api_types.h"
#include "ffx/api/include/vk/ffx_api_vk.h"
#include "ffx/framegeneration/include/ffx_framegeneration.h"

#include "assets_gen.h"

extern "C" const unsigned char fsr4fg_assets[];   // (assets.S: build/gen/assets.bin)

namespace {

constexpr uint64_t VERSION_ID = 0xf600000001000002ull;   // (AMD's id for 4.0.2, so a version override naming it lands here)
constexpr const char* VERSION_NAME = "4.0.2 (fsr4fg-vk)";
constexpr uint32_t RING = 8;        // frames a descriptor set or constants slot is left alone before it is written again
constexpr uint32_t TENSOR_BYTES = 92 * 16;

// ---------------------------------------------------------------- logging
ffxApiMessage g_message = nullptr;
uint32_t g_message_level = 0;

void say(bool error, const char* fmt, ...) {
    static const char* path = getenv("FSR4FG_LOG");
    static const bool verbose = getenv("FSR4FG_VERBOSE") != nullptr;
    if (!error && !verbose && !path && !g_message) return;
    char text[1024];
    va_list a;
    va_start(a, fmt);
    vsnprintf(text, sizeof text, fmt, a);
    va_end(a);
    if (error || verbose) fprintf(stderr, "fsr4fg: %s\n", text);
    if (path) {
        static std::mutex mu;
        std::lock_guard lock(mu);
        if (FILE* f = fopen(path, "a")) fprintf(f, "%s\n", text), fclose(f);
    }
    if (g_message && (error || g_message_level > 1)) {
        wchar_t wide[1024];
        size_t i = 0;
        for (; text[i] && i < 1023; i++) wide[i] = (unsigned char)text[i];
        wide[i] = 0;
        g_message(error ? FFX_API_MESSAGE_TYPE_ERROR : FFX_API_MESSAGE_TYPE_WARNING, wide);
    }
}

// ---------------------------------------------------------------- the device's functions
#define DEVICE_FUNCS(X)                                                                                                \
    X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) \
    X(MapMemory) X(UnmapMemory) X(CreateImage) X(DestroyImage) X(GetImageMemoryRequirements) X(BindImageMemory)        \
    X(CreateImageView) X(DestroyImageView) X(CreateSampler) X(DestroySampler) X(CreateShaderModule)                    \
    X(DestroyShaderModule) X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) X(CreatePipelineLayout)          \
    X(DestroyPipelineLayout) X(CreateComputePipelines) X(DestroyPipeline) X(CreateDescriptorPool)                      \
    X(DestroyDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) X(CmdBindPipeline)                      \
    X(CmdBindDescriptorSets) X(CmdDispatch) X(CmdPipelineBarrier) X(CmdCopyImage) X(CmdCopyBuffer)                     \
    X(CmdClearColorImage) X(CmdFillBuffer)
struct Vk {
#define DECL(n) PFN_vk##n n = nullptr;
    DEVICE_FUNCS(DECL)
#undef DECL
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageView view64 = VK_NULL_HANDLE;   // (the splat images: the R64_UINT view their writer's atomics need)
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
};
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* map = nullptr;
};

/// A pass: its pipeline and the sets it is bound with. One set when everything it reads is ours; a ring of them when it
/// reads one of the caller's images, which may be a different image every frame.
struct Pass {
    const GenPass* gen = nullptr;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets;
    bool external = false;
    bool dynamic = false;   // has a per-frame constants block (bound with a dynamic offset)
    uint32_t dynamic_range = 0;
    uint32_t groups[3]{1, 1, 1};
};

// ---------------------------------------------------------------- camera matrices
// Bit for bit as AMD's DLL computes them (fitted from traces of its constants, every element of every frame, in the
// four depth modes and at two aspects): single precision throughout, products accumulated in index order with each
// step rounded, and the cofactor inverse below. Its rounding reaches the output: the forward motion pass compares
// these matrices' reprojection with the motion vectors to within 2 pixels. (Build with -ffp-contract=off.)
struct Mat4 {
    float m[4][4];
};
Mat4 identity() {
    Mat4 r{};
    for (int i = 0; i < 4; i++) r.m[i][i] = 1.0f;
    return r;
}
Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float s = a.m[i][0] * b.m[0][j];
            for (int k = 1; k < 4; k++) s = s + a.m[i][k] * b.m[k][j];
            r.m[i][j] = s;
        }
    return r;
}
/// The cofactor inverse (the well-known MESA gluInvertMatrix form, term for term), scaled by 1/det. A singular matrix
/// gives infinities and NaNs, as it would in AMD's.
Mat4 inverse(const Mat4& in) {
    const float* m = &in.m[0][0];
    float inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    const float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    const float rdet = 1.0f / det;
    Mat4 out;
    for (int i = 0; i < 16; i++) (&out.m[0][0])[i] = inv[i] * rdet;
    return out;
}

// The constants blocks, as AMD's shaders declare them (the member names are theirs, from the SPIR-V's debug names).
struct CsDilate {
    float motionVectorJitterCancellation[2];
    int32_t renderSize[2];
    int32_t displaySize[2];
    int32_t dispatchSize[2];
    float motionVectorScale[2];
    float generationRectMinUV[2];
    float generationRectMaxUV[2];
    uint32_t depthInverted;
    uint32_t pad1;
    float colorRectMinUV[2];
    float colorRectMaxUV[2];
};
struct CsSplat {
    int32_t dstSize[2];
    int32_t renderSize[2];
    float projAmount;
    int32_t flowIsUv;
    float generationRectMinUV[2];
    float generationRectMaxUV[2];
    uint32_t depthInverted;
    uint32_t pad[5];
};
struct CsFwdMv {
    float prevPsToPs[16];
    float prevPsToPrev2Ps[16];
    uint32_t displaySize[2];
    uint32_t pad[2];
};
struct CsFi {
    uint32_t renderSize[2];
    uint32_t displaySize[2];
    int32_t reset;
    uint32_t backBufferTransferFunction;
    int32_t hudlessAttachedFactor;
    int32_t dispatchFlags;
    float minMaxLuminance[2];
    float generationRectMinUV[2];
    float generationRectMaxUV[2];
    uint32_t pad1[2];
    float debugBarColor[3];
    uint32_t pad2;
};
static_assert(sizeof(CsDilate) == 80 && sizeof(CsSplat) == 64 && sizeof(CsFwdMv) == 144 && sizeof(CsFi) == 80);

// where each block of a frame's constants is, within its slot
enum : uint32_t { SLOT_DILATE = 0, SLOT_SPLAT_A = 256, SLOT_FWD = 512, SLOT_SPLAT_B = 768, SLOT_SPLAT_C = 1024, SLOT_FI = 1280, SLOT_BYTES = 1536 };

struct Context {
    Vk vk;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};

    uint32_t flags = 0;
    uint32_t width = 0, height = 0;          // the display size
    uint32_t third_w = 0, third_h = 0;       // a third of it: the preparation's images
    uint32_t tensors[92][2]{};
    VkFormat color_format = VK_FORMAT_UNDEFINED, history_format = VK_FORMAT_UNDEFINED;
    bool big = false;                        // the 8k build of the network

    Buffer scratch, weights, weights_staging, ubo;
    Image flow0, flow1, dilated_mv, prev_mv, dilated_depth, prev_depth, splat_cur, splat_fwd, fwd_mv, prev_color, distortion, hudless;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;

    Pass dilate, splat, fwd;
    VkDescriptorSet splat_set_cur = VK_NULL_HANDLE, splat_set_fwd = VK_NULL_HANDLE;
    std::vector<Pass> passes;

    std::vector<VkImageView> views[RING];    // the views made of the caller's images, by the slot that made them

    bool initialised = false;                // the weights uploaded, our images cleared
    uint64_t frames = 0;                     // prepared so far
    uint32_t slot = 0;
    bool prepared = false;
    uint64_t prepared_id = 0;
    bool prepare_reset = false;
    uint32_t render_w = 0, render_h = 0;
    Mat4 clip_to_world_prev = identity(), clip_to_world_prev2 = identity();
    float jitter_prev[2] = {0.0f, 0.0f};
    FfxApiResource hudless_res{};   // (the HUD-less colour of the last configure; none when its resource is empty)   // (the previous prepare's jitter; a reset keeps it, as AMD's does)
    uint32_t debug_color = 0;

    // ffxConfigure
    bool enabled = true;
    uint32_t config_flags = 0;
    FfxApiRect2D rect{};
};

uint32_t memory_type(Context& c, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c.memory.memoryTypeCount; i++)
        if ((bits >> i & 1) && (c.memory.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

bool make_buffer(Context& c, Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage, bool host) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size, ci.usage = usage;
    if (c.vk.CreateBuffer(c.device, &ci, nullptr, &b.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    c.vk.GetBufferMemoryRequirements(c.device, b.buffer, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = memory_type(c, mr.memoryTypeBits, host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX || c.vk.AllocateMemory(c.device, &ai, nullptr, &b.memory) != VK_SUCCESS) return false;
    if (c.vk.BindBufferMemory(c.device, b.buffer, b.memory, 0) != VK_SUCCESS) return false;
    b.size = size;
    if (host && c.vk.MapMemory(c.device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.map) != VK_SUCCESS) return false;
    return true;
}
void drop_buffer(Context& c, Buffer& b) {
    if (b.buffer) c.vk.DestroyBuffer(c.device, b.buffer, nullptr);
    if (b.memory) c.vk.FreeMemory(c.device, b.memory, nullptr);
    b = Buffer{};
}
/// A device-local image with a view of `format`. With splat, the image is R64_UINT and mutable, `format` is the
/// readers' R32G32_UINT view and view64 the writer's: AMD's DLL makes it R32G32_UINT and runs the 64-bit atomics on
/// that, which the spec does not allow (no STORAGE_IMAGE_ATOMIC on R32G32_UINT). The bits are the same either way.
bool make_image(Context& c, Image& im, VkFormat format, uint32_t w, uint32_t h, bool splat = false) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    if (splat) ci.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;   // (R64_UINT cannot be sampled)
    ci.imageType = VK_IMAGE_TYPE_2D, ci.format = splat ? VK_FORMAT_R64_UINT : format, ci.extent = {w, h, 1}, ci.mipLevels = 1, ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT, ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (c.vk.CreateImage(c.device, &ci, nullptr, &im.image) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    c.vk.GetImageMemoryRequirements(c.device, im.image, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size, ai.memoryTypeIndex = memory_type(c, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX || c.vk.AllocateMemory(c.device, &ai, nullptr, &im.memory) != VK_SUCCESS) return false;
    if (c.vk.BindImageMemory(c.device, im.image, im.memory, 0) != VK_SUCCESS) return false;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im.image, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (c.vk.CreateImageView(c.device, &vi, nullptr, &im.view) != VK_SUCCESS) return false;
    VkImageViewUsageCreateInfo storage_only{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    storage_only.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    vi.format = VK_FORMAT_R64_UINT, vi.pNext = &storage_only;
    if (splat && c.vk.CreateImageView(c.device, &vi, nullptr, &im.view64) != VK_SUCCESS) return false;
    im.format = format, im.width = w, im.height = h;
    return true;
}
void drop_image(Context& c, Image& im) {
    if (im.view) c.vk.DestroyImageView(c.device, im.view, nullptr);
    if (im.view64) c.vk.DestroyImageView(c.device, im.view64, nullptr);
    if (im.image) c.vk.DestroyImage(c.device, im.image, nullptr);
    if (im.memory) c.vk.FreeMemory(c.device, im.memory, nullptr);
    im = Image{};
}

// ---------------------------------------------------------------- sizes
/// The size of tensor `i` of the network for a display dimension `v`: every tensor's is one of ten functions of it.
uint32_t tensor_size(TensorRule rule, uint32_t v) {
    const uint32_t eighth = (v + 7) / 8;                 // the flow's size
    const uint32_t coarse = (eighth + 31) / 32 * 32;     // the first network's, a whole number of its 32-wide blocks
    const uint32_t even = eighth + (eighth & 1);         // the second network's half-size layers
    switch (rule) {
    case T_A0: return coarse;
    case T_A1: return coarse >> 1;
    case T_A2: return coarse >> 2;
    case T_A3: return coarse >> 3;
    case T_A4: return coarse >> 4;
    case T_F: return eighth;
    case T_G: return even;
    case T_G2: return even * 2;
    case T_Gh: return even >> 1;
    case T_D: return v;
    }
    return 0;
}
uint32_t term(const Context& c, const GenTerm& t) {
    switch (t.kind) {
    case 'k': return t.a;
    case 't': return (c.tensors[t.a][t.b] + t.c - 1) / t.c * t.d;
    case 'p': {   // a border-clearing pass: one invocation an element of the border, as its shader counts them
        const uint32_t w = c.tensors[t.a][0], h = c.tensors[t.a][1];
        const uint32_t dx = std::min(t.b - w, t.d), dy = std::min(t.c - h, t.e);
        const uint32_t w1 = 1 + w + dx;
        return (w1 + h + h * dx + w1 * dy + 31) / 32;
    }
    }
    return 1;
}

// ---------------------------------------------------------------- passes
VkDescriptorType descriptor_type(const GenBind& b) {
    switch (b.kind) {
    case KIND_SAMPLER: return VK_DESCRIPTOR_TYPE_SAMPLER;
    case KIND_SAMPLED: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case KIND_STORAGE: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case KIND_UBO: return b.res == RES_UBO_TENSOR ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    case KIND_SSBO: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    return VK_DESCRIPTOR_TYPE_SAMPLER;
}
bool is_external(Res r) { return r == RES_COLOR || r == RES_OUTPUT || r == RES_MOTION || r == RES_DEPTH; }

bool make_pass(Context& c, Pass& p, const GenPass& gen, uint32_t dynamic_range) {
    p.gen = &gen;
    p.dynamic_range = dynamic_range;
    VkDescriptorSetLayoutBinding binds[12]{};
    for (uint32_t i = 0; i < gen.bind_count; i++) {
        binds[i].binding = gen.binds[i].binding;
        binds[i].descriptorType = descriptor_type(gen.binds[i]);
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        p.external |= is_external(gen.binds[i].res);
        p.dynamic |= binds[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    }
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = gen.bind_count, li.pBindings = binds;
    if (c.vk.CreateDescriptorSetLayout(c.device, &li, nullptr, &p.set_layout) != VK_SUCCESS) return false;
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1, pi.pSetLayouts = &p.set_layout;
    if (c.vk.CreatePipelineLayout(c.device, &pi, nullptr, &p.layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = gen.size, mi.pCode = (const uint32_t*)(fsr4fg_assets + gen.offset);
    VkShaderModule module;
    if (c.vk.CreateShaderModule(c.device, &mi, nullptr, &module) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT, ci.stage.module = module, ci.stage.pName = gen.entry;
    ci.layout = p.layout;
    // (the network's passes are written for 32-wide subgroups, and AMD makes them so: RADV would otherwise be free to run
    // them 64 wide, which gives wrong results from the first pass that depends on it)
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo wave{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    wave.requiredSubgroupSize = 32;
    if (strncmp(gen.entry, "mlfi_", 5) == 0) ci.stage.pNext = &wave;
    const VkResult r = c.vk.CreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipeline);
    c.vk.DestroyShaderModule(c.device, module, nullptr);
    if (r != VK_SUCCESS) {
        say(true, "the pipeline of %s was not made (VkResult %d)", gen.entry, (int)r);
        return false;
    }
    for (int d = 0; d < 3; d++) p.groups[d] = term(c, gen.groups[d]);
    return true;
}
bool allocate_sets(Context& c, Pass& p, uint32_t count) {
    p.sets.resize(count);
    std::vector<VkDescriptorSetLayout> layouts(count, p.set_layout);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = c.pool, ai.descriptorSetCount = count, ai.pSetLayouts = layouts.data();
    return c.vk.AllocateDescriptorSets(c.device, &ai, p.sets.data()) == VK_SUCCESS;
}
void drop_pass(Context& c, Pass& p) {
    if (p.pipeline) c.vk.DestroyPipeline(c.device, p.pipeline, nullptr);
    if (p.layout) c.vk.DestroyPipelineLayout(c.device, p.layout, nullptr);
    if (p.set_layout) c.vk.DestroyDescriptorSetLayout(c.device, p.set_layout, nullptr);
    p = Pass{};
}

/// What a frame's passes are given of the caller's: the views of its images (null where the pass has none).
struct External {
    VkImageView color = VK_NULL_HANDLE, output = VK_NULL_HANDLE, motion = VK_NULL_HANDLE, depth = VK_NULL_HANDLE;
    VkImageView hudless = VK_NULL_HANDLE;   // (the caller's HUD-less colour, if configured; else the port's 1x1 stand-in)
};
/// What the splat pass is bound to: it runs three times a frame over two sets of images.
struct SplatImages {
    const Image* flow;
    const Image* depth;
    const Image* out;
};

/// Writes one set of a pass. `only_external` leaves out what does not change (a ring set written again for a new frame).
void write_set(Context& c, const Pass& p, VkDescriptorSet set, const External& ext, const SplatImages* splat, bool only_external) {
    VkWriteDescriptorSet writes[12]{};
    VkDescriptorImageInfo images[12]{};
    VkDescriptorBufferInfo buffers[12]{};
    uint32_t n = 0;
    for (uint32_t i = 0; i < p.gen->bind_count; i++) {
        const GenBind& b = p.gen->binds[i];
        if (only_external && !is_external(b.res)) continue;
        VkWriteDescriptorSet& w = writes[n];
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set, w.dstBinding = b.binding, w.descriptorCount = 1, w.descriptorType = descriptor_type(b);
        const Image* im = nullptr;
        VkImageView view = VK_NULL_HANDLE;
        switch (b.res) {
        case RES_SCRATCH: buffers[n] = {c.scratch.buffer, 0, VK_WHOLE_SIZE}; break;
        case RES_WEIGHTS: buffers[n] = {c.weights.buffer, 0, VK_WHOLE_SIZE}; break;
        case RES_UBO_TENSOR: buffers[n] = {c.ubo.buffer, 0, TENSOR_BYTES}; break;
        case RES_UBO_FI: buffers[n] = {c.ubo.buffer, 0, sizeof(CsFi)}; break;
        case RES_UBO_PREP: buffers[n] = {c.ubo.buffer, 0, p.dynamic_range}; break;
        case RES_SAMPLER: images[n].sampler = c.sampler; break;
        case RES_COLOR_PREV: im = &c.prev_color; break;
        case RES_FLOW0: im = &c.flow0; break;
        case RES_FLOW1: im = &c.flow1; break;
        case RES_SPLAT0: im = &c.splat_fwd; break;
        case RES_SPLAT1: im = &c.splat_cur; break;
        case RES_HUDLESS: if (ext.hudless) view = ext.hudless; else im = &c.hudless; break;
        case RES_DISTORTION: im = &c.distortion; break;
        case RES_DILATED_MV: im = &c.dilated_mv; break;
        case RES_DILATED_DEPTH: im = &c.dilated_depth; break;
        case RES_PREV_MV: im = &c.prev_mv; break;
        case RES_PREV_DEPTH: im = &c.prev_depth; break;
        case RES_SPLAT_1TO0: im = &c.splat_cur; break;
        case RES_FWD_MV: im = &c.fwd_mv; break;
        case RES_SPLAT_FLOW: im = splat->flow; break;
        case RES_SPLAT_DEPTH: im = splat->depth; break;
        case RES_SPLAT_OUT: im = splat->out; break;
        case RES_COLOR: view = ext.color; break;
        case RES_OUTPUT: view = ext.output; break;
        case RES_MOTION: view = ext.motion; break;
        case RES_DEPTH: view = ext.depth; break;
        default: break;
        }
        if (b.kind == KIND_SAMPLED || b.kind == KIND_STORAGE) {
            images[n].imageView = im ? (b.res == RES_SPLAT_OUT && im->view64 ? im->view64 : im->view) : view;
            images[n].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
        if (b.kind == KIND_UBO || b.kind == KIND_SSBO) w.pBufferInfo = &buffers[n];
        else w.pImageInfo = &images[n];
        n++;
    }
    if (n) c.vk.UpdateDescriptorSets(c.device, n, writes, 0, nullptr);
}

void barrier(Context& c, VkCommandBuffer cmd) {
    VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    m.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    m.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    c.vk.CmdPipelineBarrier(cmd, stages, stages, 0, 1, &m, 0, nullptr, 0, nullptr);
}
void run(Context& c, VkCommandBuffer cmd, const Pass& p, VkDescriptorSet set, uint32_t dynamic_offset, const uint32_t* groups = nullptr) {
    barrier(c, cmd);
    c.vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    c.vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, p.dynamic ? 1 : 0, &dynamic_offset);
    const uint32_t* g = groups ? groups : p.groups;
    c.vk.CmdDispatch(cmd, g[0], g[1], g[2]);
}
void image_barrier(Context& c, VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = from, b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image, b.subresourceRange = {aspect, 0, 1, 0, 1};
    c.vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}
void clear(Context& c, VkCommandBuffer cmd, const Image& im) {
    barrier(c, cmd);
    const VkClearColorValue zero{};
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    c.vk.CmdClearColorImage(cmd, im.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
}
void copy(Context& c, VkCommandBuffer cmd, VkImage from, const Image& to) {
    barrier(c, cmd);
    VkImageCopy r{};
    r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.extent = {to.width, to.height, 1};
    c.vk.CmdCopyImage(cmd, from, VK_IMAGE_LAYOUT_GENERAL, to.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
}

// ---------------------------------------------------------------- the caller's images
VkImageLayout layout_of(uint32_t state) {
    switch (state) {
    case FFX_API_RESOURCE_STATE_COMPUTE_READ:
    case FFX_API_RESOURCE_STATE_PIXEL_READ:
    case FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ: return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case FFX_API_RESOURCE_STATE_COPY_SRC: return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case FFX_API_RESOURCE_STATE_COPY_DEST: return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    case FFX_API_RESOURCE_STATE_PRESENT: return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    case FFX_API_RESOURCE_STATE_RENDER_TARGET: return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case FFX_API_RESOURCE_STATE_DEPTH_ATTACHMENT: return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    default: return VK_IMAGE_LAYOUT_GENERAL;   // (common, generic read, unordered access: as AMD's Vulkan backend has them)
    }
}
/// The Vulkan format of a resource the caller describes by AMD's surface format (AMD's header only maps the other way).
VkFormat vk_format(const FfxApiResourceDescription& d) {
    static const VkFormat depth[]{VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT,
                                  VK_FORMAT_X8_D24_UNORM_PACK32, VK_FORMAT_D16_UNORM_S8_UINT};
    static const VkFormat color[]{
        VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_A2B10G10R10_UNORM_PACK32,
        VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32_SFLOAT,
        VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_UNORM, VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R8G8B8A8_SNORM,
        VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8G8_UINT,
        VK_FORMAT_R32_UINT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32};
    if (d.usage & FFX_API_RESOURCE_USAGE_DEPTHTARGET)
        for (VkFormat f : depth)
            if (ffxApiGetSurfaceFormatVK(f) == d.format) return f;
    for (VkFormat f : color)
        if (ffxApiGetSurfaceFormatVK(f) == d.format) return f;
    return VK_FORMAT_UNDEFINED;
}
bool is_depth(VkFormat f) {
    return f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT ||
           f == VK_FORMAT_X8_D24_UNORM_PACK32 || f == VK_FORMAT_D16_UNORM_S8_UINT;
}
struct Borrowed {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
};
/// A view of one of the caller's images, kept until its slot of the ring comes round again, and the image brought to
/// GENERAL for the passes when the caller has it in another layout (given back by `give_back`).
bool borrow(Context& c, VkCommandBuffer cmd, const FfxApiResource& r, Borrowed& out, const char* what) {
    if (!r.resource) {
        say(true, "%s: no image given", what);
        return false;
    }
    const VkFormat format = vk_format(r.description);
    if (format == VK_FORMAT_UNDEFINED) {
        say(true, "%s: surface format %u has no Vulkan format here", what, r.description.format);
        return false;
    }
    out.image = (VkImage)r.resource;
    out.aspect = is_depth(format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    out.layout = layout_of(r.state);
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = out.image, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = format;
    vi.subresourceRange = {out.aspect, 0, 1, 0, 1};
    if (c.vk.CreateImageView(c.device, &vi, nullptr, &out.view) != VK_SUCCESS) {
        say(true, "%s: no view of the image (format %d)", what, (int)format);
        return false;
    }
    c.views[c.slot].push_back(out.view);
    if (out.layout != VK_IMAGE_LAYOUT_GENERAL) image_barrier(c, cmd, out.image, out.aspect, out.layout, VK_IMAGE_LAYOUT_GENERAL);
    return true;
}
void give_back(Context& c, VkCommandBuffer cmd, const Borrowed& b) {
    if (b.image && b.layout != VK_IMAGE_LAYOUT_GENERAL) image_barrier(c, cmd, b.image, b.aspect, VK_IMAGE_LAYOUT_GENERAL, b.layout);
}

// ---------------------------------------------------------------- create and destroy
void destroy(Context* c) {
    if (!c) return;
    if (c->device) {
        for (auto& ring : c->views)
            for (VkImageView v : ring) c->vk.DestroyImageView(c->device, v, nullptr);
        drop_pass(*c, c->dilate), drop_pass(*c, c->splat), drop_pass(*c, c->fwd);
        for (Pass& p : c->passes) drop_pass(*c, p);
        if (c->pool) c->vk.DestroyDescriptorPool(c->device, c->pool, nullptr);
        if (c->sampler) c->vk.DestroySampler(c->device, c->sampler, nullptr);
        for (Image* im : {&c->flow0, &c->flow1, &c->dilated_mv, &c->prev_mv, &c->dilated_depth, &c->prev_depth, &c->splat_cur, &c->splat_fwd,
                          &c->fwd_mv, &c->prev_color, &c->distortion, &c->hudless})
            drop_image(*c, *im);
        for (Buffer* b : {&c->scratch, &c->weights, &c->weights_staging, &c->ubo}) drop_buffer(*c, *b);
    }
    delete c;
}

const ffxApiHeader* find(const ffxApiHeader* h, uint64_t type) {
    for (; h; h = h->pNext)
        if (h->type == type) return h;
    return nullptr;
}

ffxReturnCode_t create(ffxContext* out, ffxCreateContextDescHeader* header) {
    const auto* desc = (const ffxCreateContextDescFrameGeneration*)find(header, FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION);
    if (!desc) return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    // (the backend description: AMD's current one and the one it replaced lay out the same three members)
    const ffxApiHeader* bh = find(header, FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK);
    if (!bh) bh = find(header, FFX_API_MAKE_BACKEND_SUB_ID(FFX_API_BACKEND_ID_VK, 0x01));
    if (!bh) {
        say(true, "no Vulkan backend description with the context");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    const VkDeviceContext& dc = ((const ffxCreateBackendVKDesc*)bh)->deviceContext;
    if (!dc.vkDevice || !dc.vkPhysicalDevice || !dc.vkDeviceProcAddr) return FFX_API_RETURN_ERROR_PARAMETER;
    if (const auto* ov = (const ffxOverrideVersion*)find(header, FFX_API_DESC_TYPE_OVERRIDE_VERSION); ov && ov->versionId != VERSION_ID) {
        say(true, "version 0x%llx was asked for: this library is only %s", (unsigned long long)ov->versionId, VERSION_NAME);
        return FFX_API_RETURN_NO_PROVIDER;
    }
    if (!desc->displaySize.width || !desc->displaySize.height || desc->displaySize.width > 7680 || desc->displaySize.height > 4320) {
        say(true, "a display of %ux%u is outside what the network was built for", desc->displaySize.width, desc->displaySize.height);
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    if (desc->flags & FFX_FRAMEGENERATION_ENABLE_ASYNC_WORKLOAD_SUPPORT) say(true, "ENABLE_ASYNC_WORKLOAD_SUPPORT is not implemented and is ignored");


    Context* c = new Context;
    c->device = dc.vkDevice, c->physical = dc.vkPhysicalDevice;
#define LOAD(n) c->vk.n = (PFN_vk##n)dc.vkDeviceProcAddr(c->device, "vk" #n);
    DEVICE_FUNCS(LOAD)
#undef LOAD
#define CHECK(n) \
    if (!c->vk.n) { say(true, "the device has no vk" #n); delete c; return FFX_API_RETURN_ERROR_RUNTIME_ERROR; }
    DEVICE_FUNCS(CHECK)
#undef CHECK
    // (the one instance-level call: from the Vulkan loader the caller's device came through)
    void* loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!loader) loader = dlopen("libvulkan.so.1", RTLD_NOW);
    auto memory_properties = loader ? (PFN_vkGetPhysicalDeviceMemoryProperties)dlsym(loader, "vkGetPhysicalDeviceMemoryProperties") : nullptr;
    if (!memory_properties) {
        say(true, "no vkGetPhysicalDeviceMemoryProperties in libvulkan.so.1");
        delete c;
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    memory_properties(c->physical, &c->memory);

    c->flags = desc->flags;
    c->width = desc->displaySize.width, c->height = desc->displaySize.height;
    c->third_w = std::max(1u, c->width / 3), c->third_h = std::max(1u, c->height / 3);
    c->big = c->width > 3840 || c->height > 2160;
    c->rect = {0, 0, (int32_t)c->width, (int32_t)c->height};
    FfxApiResourceDescription back{};
    back.format = desc->backBufferFormat;
    c->color_format = vk_format(back);
    if (c->color_format == VK_FORMAT_UNDEFINED) {
        say(true, "back buffer surface format %u has no Vulkan format here", desc->backBufferFormat);
        delete c;
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    // (with a HUD-less colour, the network's picture is that one, so the previous frame's copy is in its format: AMD's
    // keeps it in the format of ffxCreateContextDescFrameGenerationHudless when that is given, traced)
    c->history_format = c->color_format;
    if (const auto* hl = (const ffxCreateContextDescFrameGenerationHudless*)find(header, FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS)) {
        FfxApiResourceDescription h{};
        h.format = hl->hudlessBackBufferFormat;
        if (vk_format(h) != VK_FORMAT_UNDEFINED) c->history_format = vk_format(h);
    }
    for (int i = 0; i < 92; i++)
        c->tensors[i][0] = tensor_size(GEN_TENSOR_RULES[i], c->width), c->tensors[i][1] = tensor_size(GEN_TENSOR_RULES[i], c->height);

    bool ok = true;
    const uint32_t flow_w = (c->width + 7) / 8, flow_h = (c->height + 7) / 8;
    ok = ok && make_buffer(*c, c->scratch, c->big ? GEN_SCRATCH_BYTES_8K : GEN_SCRATCH_BYTES_4K,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);
    ok = ok && make_buffer(*c, c->weights, GEN_WEIGHTS_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
    ok = ok && make_buffer(*c, c->weights_staging, GEN_WEIGHTS_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    ok = ok && make_buffer(*c, c->ubo, SLOT_BYTES * (RING + 1), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
    ok = ok && make_image(*c, c->flow0, VK_FORMAT_R16G16_SFLOAT, flow_w, flow_h) && make_image(*c, c->flow1, VK_FORMAT_R16G16_SFLOAT, flow_w, flow_h);
    ok = ok && make_image(*c, c->dilated_mv, VK_FORMAT_R16G16_SFLOAT, c->third_w, c->third_h) && make_image(*c, c->prev_mv, VK_FORMAT_R16G16_SFLOAT, c->third_w, c->third_h);
    ok = ok && make_image(*c, c->dilated_depth, VK_FORMAT_R32_SFLOAT, c->third_w, c->third_h) && make_image(*c, c->prev_depth, VK_FORMAT_R32_SFLOAT, c->third_w, c->third_h);
    // (the splats are 64-bit texels to their shader and two 32-bit ones to the passes that read them, as AMD makes them)
    ok = ok && make_image(*c, c->splat_cur, VK_FORMAT_R32G32_UINT, c->third_w, c->third_h, true) &&
              make_image(*c, c->splat_fwd, VK_FORMAT_R32G32_UINT, c->third_w, c->third_h, true);
    ok = ok && make_image(*c, c->fwd_mv, VK_FORMAT_R16G16_SFLOAT, c->third_w, c->third_h);
    ok = ok && make_image(*c, c->prev_color, c->history_format, c->width, c->height);
    ok = ok && make_image(*c, c->distortion, VK_FORMAT_R8G8_UNORM, 1, 1) && make_image(*c, c->hudless, VK_FORMAT_R8G8B8A8_UNORM, 1, 1);
    if (ok) {
        memcpy(c->weights_staging.map, fsr4fg_assets + GEN_WEIGHTS_OFFSET, GEN_WEIGHTS_SIZE);
        uint32_t* t = (uint32_t*)c->ubo.map;
        memset(t, 0, SLOT_BYTES);
        for (int i = 0; i < 92; i++) t[i * 4] = c->tensors[i][0], t[i * 4 + 1] = c->tensors[i][1];
        // AMD's Vulkan build gives every pass one sampler, nearest and clamped, whatever the shader calls it. The same
        // here, so the pictures are the same; FSR4FG_LINEAR=1 makes it a linear one.
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        const bool linear = getenv("FSR4FG_LINEAR") && atoi(getenv("FSR4FG_LINEAR")) > 0;
        si.magFilter = si.minFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = VK_LOD_CLAMP_NONE;
        ok = c->vk.CreateSampler(c->device, &si, nullptr, &c->sampler) == VK_SUCCESS;
    }
    if (!ok) say(true, "a buffer or image of the context was not made");

    const GenPass* gen = c->big ? GEN_PASSES_8K : GEN_PASSES_4K;
    const size_t gen_count = c->big ? std::size(GEN_PASSES_8K) : std::size(GEN_PASSES_4K);
    if (ok) {
        c->passes.resize(gen_count);
        // (the dilate pass's variant by the create flags, as AMD's DLL picks it: traced)
        const GenPass& dilate = (desc->flags & FFX_FRAMEGENERATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) ? GEN_DILATE_DISPLAY
                              : (desc->flags & FFX_FRAMEGENERATION_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) ? GEN_DILATE_JITTER : GEN_DILATE;
        ok = make_pass(*c, c->dilate, dilate, sizeof(CsDilate)) && make_pass(*c, c->splat, GEN_SPLAT, sizeof(CsSplat)) &&
             make_pass(*c, c->fwd, GEN_FWDMV, sizeof(CsFwdMv));
        for (size_t i = 0; ok && i < gen_count; i++) ok = make_pass(*c, c->passes[i], gen[i], sizeof(CsFi));
    }
    if (ok) {
        uint32_t sets = 2 + 1 + RING, counts[5]{};
        auto tally = [&](const Pass& p, uint32_t n) {
            for (uint32_t i = 0; i < p.gen->bind_count; i++) counts[p.gen->binds[i].kind] += n;
        };
        tally(c->dilate, RING), tally(c->splat, 2), tally(c->fwd, 1);
        for (const Pass& p : c->passes) sets += p.external ? RING : 1, tally(p, p.external ? RING : 1);
        const VkDescriptorPoolSize sizes[]{
            {VK_DESCRIPTOR_TYPE_SAMPLER, counts[KIND_SAMPLER] + 1},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, counts[KIND_SAMPLED] + 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, counts[KIND_STORAGE] + 1},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, counts[KIND_UBO] + 1},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, counts[KIND_UBO] + 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, counts[KIND_SSBO] + 1},
        };
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = sets, pi.poolSizeCount = (uint32_t)std::size(sizes), pi.pPoolSizes = sizes;
        ok = c->vk.CreateDescriptorPool(c->device, &pi, nullptr, &c->pool) == VK_SUCCESS;
        ok = ok && allocate_sets(*c, c->dilate, RING) && allocate_sets(*c, c->splat, 2) && allocate_sets(*c, c->fwd, 1);
        for (size_t i = 0; ok && i < gen_count; i++) ok = allocate_sets(*c, c->passes[i], c->passes[i].external ? RING : 1);
        if (!ok) say(true, "the descriptor sets were not made");
    }
    if (ok) {
        const External none{};
        const SplatImages cur{&c->dilated_mv, &c->dilated_depth, &c->splat_cur}, fwd{&c->fwd_mv, &c->prev_depth, &c->splat_fwd};
        c->splat_set_cur = c->splat.sets[0], c->splat_set_fwd = c->splat.sets[1];
        write_set(*c, c->splat, c->splat_set_cur, none, &cur, false);
        write_set(*c, c->splat, c->splat_set_fwd, none, &fwd, false);
        write_set(*c, c->fwd, c->fwd.sets[0], none, nullptr, false);
        for (Pass& p : c->passes)
            if (!p.external) write_set(*c, p, p.sets[0], none, nullptr, false);
    }
    if (!ok) {
        destroy(c);
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    // (a pass that reads the caller's images has a ring of sets, each written whole in the frame that uses it)
    c->dilate.groups[0] = (c->third_w + 7) / 8, c->dilate.groups[1] = (c->third_h + 7) / 8, c->dilate.groups[2] = 1;
    for (Pass* p : {&c->splat, &c->fwd}) p->groups[0] = c->dilate.groups[0], p->groups[1] = c->dilate.groups[1], p->groups[2] = 1;
    say(false, "context: display %ux%u, network %s (%zu passes), first tensor %ux%u, flow %ux%u, scratch %llu MB", c->width, c->height,
        c->big ? "8k" : "4k", gen_count, c->tensors[0][0], c->tensors[0][1], flow_w, flow_h, (unsigned long long)(c->scratch.size >> 20));
    *out = c;
    return FFX_API_RETURN_OK;
}

/// Once, in the first command buffer the caller hands over: the weights to the GPU, and every image of ours cleared (they
/// are read before they are written on a first frame, and must be in GENERAL).
void initialise(Context& c, VkCommandBuffer cmd) {
    if (c.initialised) return;
    c.initialised = true;
    VkBufferCopy r{0, 0, GEN_WEIGHTS_SIZE};
    c.vk.CmdCopyBuffer(cmd, c.weights_staging.buffer, c.weights.buffer, 1, &r);
    c.vk.CmdFillBuffer(cmd, c.scratch.buffer, 0, VK_WHOLE_SIZE, 0);
    for (Image* im : {&c.flow0, &c.flow1, &c.dilated_mv, &c.prev_mv, &c.dilated_depth, &c.prev_depth, &c.splat_cur, &c.splat_fwd, &c.fwd_mv,
                      &c.prev_color, &c.distortion, &c.hudless}) {
        image_barrier(c, cmd, im->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        const VkClearColorValue zero{};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        c.vk.CmdClearColorImage(cmd, im->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    }
}

void rect_uv(const Context& c, float* min_uv, float* max_uv) {
    FfxApiRect2D r = c.rect;
    if (r.width <= 0 || r.height <= 0) r = {0, 0, (int32_t)c.width, (int32_t)c.height};
    min_uv[0] = (float)r.left / (float)c.width, min_uv[1] = (float)r.top / (float)c.height;
    max_uv[0] = (float)(r.left + r.width) / (float)c.width, max_uv[1] = (float)(r.top + r.height) / (float)c.height;
}
uint8_t* slot_bytes(Context& c) { return (uint8_t*)c.ubo.map + SLOT_BYTES * (1 + c.slot); }
uint32_t slot_offset(const Context& c, uint32_t block) { return SLOT_BYTES * (1 + c.slot) + block; }

/// Clip to world for a camera, as AMD's DLL computes it (rows are vectors): the inverse of the projection, then the
/// camera's own axes and position. The projection is left-handed with the vertical field of view and the render
/// size's aspect; inverted depth swaps the planes, an infinite far plane takes the limit.
Mat4 clip_to_world(const Context& c, const ffxDispatchDescFrameGenerationPrepareV2& d) {
    const bool inverted = c.flags & FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED, infinite = c.flags & FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE;
    const float b = 1.0f / std::tan(d.cameraFovAngleVertical * 0.5f);   // (tanf: glibc's is correctly rounded, as Wine's was)
    const float a = b / ((float)d.renderSize.width / (float)d.renderSize.height);
    Mat4 p{};
    p.m[0][0] = a, p.m[1][1] = b, p.m[2][3] = 1.0f;
    if (infinite) {
        p.m[2][2] = inverted ? 0.0f : 1.0f;
        p.m[3][2] = inverted ? d.cameraNear : -d.cameraNear;
    } else {
        const float n = inverted ? d.cameraFar : d.cameraNear, f = inverted ? d.cameraNear : d.cameraFar;
        p.m[2][2] = f / (f - n), p.m[3][2] = n * f / (n - f);   // (not -m22 * n: that differs in the last bit for some planes)
    }
    Mat4 cam{};
    for (int i = 0; i < 3; i++)
        cam.m[0][i] = d.cameraRight[i], cam.m[1][i] = d.cameraUp[i], cam.m[2][i] = d.cameraForward[i], cam.m[3][i] = d.cameraPosition[i];
    cam.m[3][3] = 1.0f;
    return mul(inverse(p), cam);
}

ffxReturnCode_t prepare(Context& c, const ffxDispatchDescFrameGenerationPrepareV2& d) {
    VkCommandBuffer cmd = (VkCommandBuffer)d.commandList;
    if (!cmd || !d.renderSize.width || !d.renderSize.height) return FFX_API_RETURN_ERROR_PARAMETER;
    initialise(c, cmd);
    c.slot = (uint32_t)(c.frames % RING);
    for (VkImageView v : c.views[c.slot]) c.vk.DestroyImageView(c.device, v, nullptr);
    c.views[c.slot].clear();

    Borrowed motion, depth;
    if (!borrow(c, cmd, d.motionVectors, motion, "prepare: motion vectors") || !borrow(c, cmd, d.depth, depth, "prepare: depth")) {
        give_back(c, cmd, motion);
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    c.render_w = d.renderSize.width, c.render_h = d.renderSize.height;
    const uint32_t inverted = (c.flags & FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED) ? 1 : 0;
    float min_uv[2], max_uv[2];
    rect_uv(c, min_uv, max_uv);

    uint8_t* slot = slot_bytes(c);
    CsDilate dil{};
    dil.renderSize[0] = c.render_w, dil.renderSize[1] = c.render_h;
    dil.displaySize[0] = c.width, dil.displaySize[1] = c.height;
    dil.dispatchSize[0] = c.third_w, dil.dispatchSize[1] = c.third_h;
    dil.motionVectorScale[0] = d.motionVectorScale.x, dil.motionVectorScale[1] = d.motionVectorScale.y;
    if (c.flags & FFX_FRAMEGENERATION_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) {
        // (as AMD's: the jitter that went from the previous frame to this one, in the motion vectors' own units)
        dil.motionVectorJitterCancellation[0] = (c.jitter_prev[0] - d.jitterOffset.x) / d.motionVectorScale.x;
        dil.motionVectorJitterCancellation[1] = (c.jitter_prev[1] - d.jitterOffset.y) / d.motionVectorScale.y;
    }
    c.jitter_prev[0] = d.jitterOffset.x, c.jitter_prev[1] = d.jitterOffset.y;
    dil.depthInverted = inverted;
    for (int i = 0; i < 2; i++)
        dil.generationRectMinUV[i] = dil.colorRectMinUV[i] = min_uv[i], dil.generationRectMaxUV[i] = dil.colorRectMaxUV[i] = max_uv[i];
    memcpy(slot + SLOT_DILATE, &dil, sizeof dil);

    auto splat = [&](uint32_t block, float amount, int uv) {
        CsSplat s{};
        s.dstSize[0] = c.third_w, s.dstSize[1] = c.third_h;
        s.renderSize[0] = c.render_w, s.renderSize[1] = c.render_h;
        s.projAmount = amount, s.flowIsUv = uv, s.depthInverted = inverted;
        for (int i = 0; i < 2; i++) s.generationRectMinUV[i] = min_uv[i], s.generationRectMaxUV[i] = max_uv[i];
        memcpy(slot + block, &s, sizeof s);
    };
    splat(SLOT_SPLAT_A, 1.0f, 0);   // this frame's motion, all the way: where each pixel was
    splat(SLOT_SPLAT_B, 0.5f, 1);   // the previous frame's motion forward, to the midpoint
    splat(SLOT_SPLAT_C, 0.5f, 0);   // this frame's motion back, to the midpoint

    // The cameras: prevPsToPs is the inverse of this frame's clip space to the previous one's, prevPsToPrev2Ps the
    // previous one's to the one before. Histories that do not exist yet are the identity, as AMD starts them.
    const Mat4 m = clip_to_world(c, d);
    const Mat4 a = inverse(mul(m, inverse(c.clip_to_world_prev))), b = mul(c.clip_to_world_prev, inverse(c.clip_to_world_prev2));
    CsFwdMv fw{};
    memcpy(fw.prevPsToPs, a.m, 64), memcpy(fw.prevPsToPrev2Ps, b.m, 64);
    fw.displaySize[0] = c.width, fw.displaySize[1] = c.height;
    memcpy(slot + SLOT_FWD, &fw, sizeof fw);
    c.clip_to_world_prev2 = c.clip_to_world_prev, c.clip_to_world_prev = m;

    External ext;
    ext.motion = motion.view, ext.depth = depth.view;
    VkDescriptorSet dilate_set = c.dilate.sets[c.slot];
    write_set(c, c.dilate, dilate_set, ext, nullptr, false);

    run(c, cmd, c.dilate, dilate_set, slot_offset(c, SLOT_DILATE));
    clear(c, cmd, c.splat_cur);
    run(c, cmd, c.splat, c.splat_set_cur, slot_offset(c, SLOT_SPLAT_A));
    run(c, cmd, c.fwd, c.fwd.sets[0], slot_offset(c, SLOT_FWD));
    clear(c, cmd, c.splat_fwd);
    run(c, cmd, c.splat, c.splat_set_fwd, slot_offset(c, SLOT_SPLAT_B));
    clear(c, cmd, c.splat_cur);
    run(c, cmd, c.splat, c.splat_set_cur, slot_offset(c, SLOT_SPLAT_C));
    copy(c, cmd, c.dilated_depth.image, c.prev_depth);
    copy(c, cmd, c.dilated_mv.image, c.prev_mv);
    barrier(c, cmd);
    give_back(c, cmd, motion), give_back(c, cmd, depth);

    c.prepared = true, c.prepared_id = d.frameID, c.prepare_reset = d.reset;
    c.frames++;
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t generate(Context& c, const ffxDispatchDescFrameGeneration& d) {
    VkCommandBuffer cmd = (VkCommandBuffer)d.commandList;
    if (!cmd) return FFX_API_RETURN_ERROR_PARAMETER;
    if (d.numGeneratedFrames != 1) {
        say(true, "%u frames asked for an interval: one is what this library makes", d.numGeneratedFrames);
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    if (!c.prepared || c.prepared_id != d.frameID) {
        say(true, "frame %llu was not prepared (ffxDispatchDescFrameGenerationPrepareV2 comes first)", (unsigned long long)d.frameID);
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    c.prepared = false;
    if (d.generationRect.width > 0 && d.generationRect.height > 0) c.rect = d.generationRect;
    Borrowed color, output, hudless;
    const bool has_hudless = c.hudless_res.resource != nullptr;
    if (!borrow(c, cmd, d.presentColor, color, "generate: present colour") || !borrow(c, cmd, d.outputs[0], output, "generate: output") ||
        (has_hudless && !borrow(c, cmd, c.hudless_res, hudless, "generate: HUD-less colour"))) {
        give_back(c, cmd, color), give_back(c, cmd, output);
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    // (the tear lines' colour of the debug view: AMD's sequence, a step a frame)
    static const float bar[7][3]{{0.0f, 1.0f, 1.0f}, {1.0f, 0.42f, 0.0f}, {0.0f, 0.16f, 1.0f}, {0.74f, 1.0f, 0.0f},
                                 {0.68f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.1f}, {1.0f, 1.0f, 0.48f}};
    CsFi fi{};
    fi.renderSize[0] = c.render_w, fi.renderSize[1] = c.render_h;
    fi.displaySize[0] = c.width, fi.displaySize[1] = c.height;
    fi.reset = (d.reset || c.prepare_reset) ? 1 : 0;
    fi.backBufferTransferFunction = d.backbufferTransferFunction;
    fi.dispatchFlags = c.config_flags & (FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_TEAR_LINES | FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_RESET_INDICATORS);
    fi.minMaxLuminance[0] = d.minMaxLuminance[0], fi.minMaxLuminance[1] = d.minMaxLuminance[1];
    fi.hudlessAttachedFactor = has_hudless ? 1 : 0;
    rect_uv(c, fi.generationRectMinUV, fi.generationRectMaxUV);
    memcpy(fi.debugBarColor, bar[c.debug_color], sizeof fi.debugBarColor);
    c.debug_color = (c.debug_color + 1) % 7;
    memcpy(slot_bytes(c) + SLOT_FI, &fi, sizeof fi);

    External ext;
    // (as AMD's, traced: with a HUD-less colour, the network's picture of this frame (r_input_rgb1) is the HUD-less one
    // and the UI input of the last pass (r_hudless_ui) is the presented colour, HUD and all)
    ext.color = has_hudless ? hudless.view : color.view, ext.output = output.view, ext.hudless = has_hudless ? color.view : VK_NULL_HANDLE;
    const uint32_t offset = slot_offset(c, SLOT_FI);
    for (Pass& p : c.passes) {
        VkDescriptorSet set = p.sets[p.external ? c.slot : 0];
        if (p.external) write_set(c, p, set, ext, nullptr, false);
        run(c, cmd, p, set, offset);
    }
    copy(c, cmd, has_hudless ? hudless.image : color.image, c.prev_color);
    barrier(c, cmd);
    give_back(c, cmd, color), give_back(c, cmd, output);
    if (has_hudless) give_back(c, cmd, hudless);
    return FFX_API_RETURN_OK;
}

} // namespace

// ---------------------------------------------------------------- the FSR API
extern "C" {

FFX_API_ENTRY ffxReturnCode_t ffxCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks*) {
    if (!context || !desc) return FFX_API_RETURN_ERROR_PARAMETER;
    *context = nullptr;
    return create(context, desc);
}

FFX_API_ENTRY ffxReturnCode_t ffxDestroyContext(ffxContext* context, const ffxAllocationCallbacks*) {
    if (!context || !*context) return FFX_API_RETURN_ERROR_PARAMETER;
    destroy((Context*)*context);
    *context = nullptr;
    return FFX_API_RETURN_OK;
}

FFX_API_ENTRY ffxReturnCode_t ffxConfigure(ffxContext* context, const ffxConfigureDescHeader* desc) {
    if (!desc) return FFX_API_RETURN_ERROR_PARAMETER;
    if (desc->type == FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG) {
        const auto* d = (const ffxConfigureDescGlobalDebug*)desc;
        g_message = d->fpMessage, g_message_level = d->debugLevel;
        return FFX_API_RETURN_OK;
    }
    if (desc->type == FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1) {
        const auto* d = (const ffxApiHeader*)desc;
        struct Debug1 { ffxApiHeader header; ffxApiMessage fpMessage; uint32_t debugLevel; };
        g_message = ((const Debug1*)d)->fpMessage, g_message_level = ((const Debug1*)d)->debugLevel;
        return FFX_API_RETURN_OK;
    }
    if (!context || !*context) return FFX_API_RETURN_ERROR_PARAMETER;
    Context& c = *(Context*)*context;
    switch (desc->type) {
    case FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION: {
        // (no swapchain of ours to tell: the caller presents, with or without FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY)
        const auto* d = (const ffxConfigureDescFrameGeneration*)desc;
        c.enabled = d->frameGenerationEnabled, c.config_flags = d->flags;
        if (d->generationRect.width > 0 && d->generationRect.height > 0) c.rect = d->generationRect;
        c.hudless_res = d->HUDLessColor;   // (as AMD's: hudlessAttachedFactor 1 and the image bound to the last pass, traced)
        return FFX_API_RETURN_OK;
    }
    case FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_KEYVALUE: return FFX_API_RETURN_OK;
    case FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_REGISTERDISTORTIONRESOURCE:
        // (AMD's 4.0.2 refuses it with ERROR_PARAMETER and changes nothing: traced. Its dilate shader binds a distortion
        // field it never gets.)
        say(true, "a distortion field is not supported by FSR Frame Generation 4.0.2 (AMD's refuses it too)");
        return FFX_API_RETURN_ERROR_PARAMETER;
    default: return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
}

FFX_API_ENTRY ffxReturnCode_t ffxQuery(ffxContext* context, ffxQueryDescHeader* desc) {
    if (!desc) return FFX_API_RETURN_ERROR_PARAMETER;
    switch (desc->type) {
    case FFX_API_QUERY_DESC_TYPE_GET_VERSIONS: {
        auto* q = (ffxQueryDescGetVersions*)desc;
        if (q->createDescType != FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION || !q->outputCount) return FFX_API_RETURN_NO_PROVIDER;
        const uint64_t capacity = *q->outputCount;
        *q->outputCount = 1;
        if (capacity >= 1) {
            if (q->versionIds) q->versionIds[0] = VERSION_ID;
            if (q->versionNames) q->versionNames[0] = VERSION_NAME;
        }
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION: {
        auto* q = (ffxQueryGetProviderVersion*)desc;
        q->versionId = VERSION_ID, q->versionName = VERSION_NAME;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE: {
        auto* q = (ffxQueryDescFrameGenerationGetGPUMemoryUsage*)desc;
        if (!context || !*context || !q->gpuMemoryUsageFrameGeneration) return FFX_API_RETURN_ERROR_PARAMETER;
        const Context& c = *(const Context*)*context;
        q->gpuMemoryUsageFrameGeneration->totalUsageInBytes =
            c.scratch.size + c.weights.size + (uint64_t)c.width * c.height * 8 + (uint64_t)c.third_w * c.third_h * 44;
        q->gpuMemoryUsageFrameGeneration->aliasableUsageInBytes = c.scratch.size;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE_V2: {
        auto* q = (ffxQueryDescFrameGenerationGetGPUMemoryUsageV2*)desc;
        if (!q->gpuMemoryUsageFrameGeneration) return FFX_API_RETURN_ERROR_PARAMETER;
        const bool big = q->displaySize.width > 3840 || q->displaySize.height > 2160;
        const uint64_t scratch = big ? GEN_SCRATCH_BYTES_8K : GEN_SCRATCH_BYTES_4K;
        q->gpuMemoryUsageFrameGeneration->totalUsageInBytes =
            scratch + GEN_WEIGHTS_SIZE + (uint64_t)q->displaySize.width * q->displaySize.height * 8 + (uint64_t)q->displaySize.width * q->displaySize.height * 5;
        q->gpuMemoryUsageFrameGeneration->aliasableUsageInBytes = scratch;
        return FFX_API_RETURN_OK;
    }
    default: return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
}

FFX_API_ENTRY ffxReturnCode_t ffxDispatch(ffxContext* context, const ffxDispatchDescHeader* desc) {
    if (!context || !*context || !desc) return FFX_API_RETURN_ERROR_PARAMETER;
    Context& c = *(Context*)*context;
    switch (desc->type) {
    case FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2: return prepare(c, *(const ffxDispatchDescFrameGenerationPrepareV2*)desc);
    case FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE: {
        // The older preparation: the same members up to the motion vectors, the camera in a description chained to it
        // (without one the camera is taken as still at the origin, which is what AMD's own does with no camera given).
        struct Old {
            ffxDispatchDescHeader header;
            uint64_t frameID;
            uint32_t flags;
            void* commandList;
            FfxApiDimensions2D renderSize;
            FfxApiFloatCoords2D jitterOffset, motionVectorScale;
            float frameTimeDelta;
            bool unused_reset;
            float cameraNear, cameraFar, cameraFovAngleVertical, viewSpaceToMetersFactor;
            FfxApiResource depth, motionVectors;
        };
        struct Camera {
            ffxConfigureDescHeader header;
            float cameraPosition[3], cameraUp[3], cameraRight[3], cameraForward[3];
        };
        const Old& o = *(const Old*)desc;
        ffxDispatchDescFrameGenerationPrepareV2 d{};
        d.frameID = o.frameID, d.flags = o.flags, d.commandList = o.commandList, d.renderSize = o.renderSize;
        d.jitterOffset = o.jitterOffset, d.motionVectorScale = o.motionVectorScale, d.frameTimeDelta = o.frameTimeDelta;
        d.cameraNear = o.cameraNear, d.cameraFar = o.cameraFar, d.cameraFovAngleVertical = o.cameraFovAngleVertical;
        d.viewSpaceToMetersFactor = o.viewSpaceToMetersFactor, d.depth = o.depth, d.motionVectors = o.motionVectors;
        d.cameraRight[0] = d.cameraUp[1] = d.cameraForward[2] = 1.0f;
        if (const auto* cam = (const Camera*)find(desc->pNext, FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO)) {
            memcpy(d.cameraPosition, cam->cameraPosition, 12), memcpy(d.cameraUp, cam->cameraUp, 12);
            memcpy(d.cameraRight, cam->cameraRight, 12), memcpy(d.cameraForward, cam->cameraForward, 12);
        }
        return prepare(c, d);
    }
    case FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION: return generate(c, *(const ffxDispatchDescFrameGeneration*)desc);
    default: return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
}

} // extern "C"
