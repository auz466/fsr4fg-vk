// Runs an FSR API frame generation provider (AMD's Windows DLL under Wine, or the native port) on a synthetic scene with
// every device-level Vulkan call it makes logged, and its images dumped. Built twice: for Windows (mingw) and for Linux.
//
//   trace <provider library> <out dir> [W H RW RH frames dumpFrom]
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#endif

// AMD's headers mark the entry points dllexport; here they are only called through pointers.
#ifndef _WIN32
#define __declspec(x)
#endif
#include "ffx/api/include/ffx_api.h"
#include "ffx/api/include/ffx_api_types.h"
#include "ffx/api/include/vk/ffx_api_vk.h"
#include "ffx/framegeneration/include/ffx_framegeneration.h"

// ---------------------------------------------------------------- Vulkan, loaded by hand
#define INSTANCE_FUNCS(X)                                                                                              \
    X(vkCreateInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)                                 \
    X(vkGetPhysicalDeviceFeatures2) X(vkEnumerateDeviceExtensionProperties) X(vkCreateDevice)                          \
    X(vkGetDeviceProcAddr) X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceQueueFamilyProperties)
#define DEVICE_FUNCS(X)                                                                                                \
    X(vkGetDeviceQueue) X(vkCreateCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer)                     \
    X(vkEndCommandBuffer) X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkDeviceWaitIdle) X(vkResetCommandBuffer)              \
    X(vkCreateImage) X(vkGetImageMemoryRequirements) X(vkAllocateMemory) X(vkBindImageMemory) X(vkCreateBuffer)        \
    X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkMapMemory) X(vkCmdCopyBufferToImage)                    \
    X(vkCmdCopyImageToBuffer) X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkDestroyBuffer) X(vkFreeMemory)            \
    X(vkCreateImageView) X(vkCreateSampler) X(vkCreateShaderModule) X(vkCreateDescriptorSetLayout)                     \
    X(vkCreatePipelineLayout) X(vkCreateComputePipelines) X(vkCreateDescriptorPool) X(vkAllocateDescriptorSets)        \
    X(vkUpdateDescriptorSets) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdDispatch)                         \
    X(vkCmdDispatchIndirect) X(vkCmdCopyImage) X(vkCmdClearColorImage) X(vkCmdFillBuffer) X(vkCmdPushConstants)        \
    X(vkCreateBufferView) X(vkUnmapMemory) X(vkFlushMappedMemoryRanges)
#define DECL(n) static PFN_##n n;
INSTANCE_FUNCS(DECL)
DEVICE_FUNCS(DECL)
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;

static FILE* g_log;
static std::string g_out;
static void logf(const char* fmt, ...) {
    va_list a;
    va_start(a, fmt);
    vfprintf(g_log, fmt, a);
    va_end(a);
    fputc('\n', g_log);
    fflush(g_log);
}
[[noreturn]] static void die(const char* fmt, ...) {
    va_list a;
    va_start(a, fmt);
    vfprintf(stderr, fmt, a);
    va_end(a);
    fputc('\n', stderr);
    exit(1);
}

static void* load_lib(const char* path) {
#ifdef _WIN32
    return (void*)LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}
static void* sym(void* lib, const char* name) {
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

// ---------------------------------------------------------------- what the provider made, by handle
struct ImageInfo {
    int id;
    VkFormat format;
    VkExtent3D extent;
    uint32_t mips;
    VkImageUsageFlags usage;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::string name;
    bool ours = false;
};
struct BufferInfo {
    int id;
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    uint64_t memory = 0;
    VkDeviceSize offset = 0;
    std::string name;
};
struct ViewInfo {
    int id;
    uint64_t image;
    uint32_t mip;
    VkImageViewCreateInfo ci{};
    VkImageView as_rg32 = VK_NULL_HANDLE;
};
struct Write {
    VkDescriptorType type;
    uint64_t a = 0;   // view or buffer
    uint64_t b = 0;   // sampler
    VkDeviceSize offset = 0, range = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};
struct PipeInfo {
    int id;
    int module;
    std::string entry;
    uint64_t layout;
};
static std::map<uint64_t, ImageInfo> g_images;
static std::map<uint64_t, BufferInfo> g_buffers;
static std::map<uint64_t, ViewInfo> g_views;
static std::map<uint64_t, void*> g_mapped;
static std::map<uint64_t, int> g_modules, g_samplers, g_setlayouts, g_pipelayouts, g_sets, g_memories;
static std::map<uint64_t, PipeInfo> g_pipes;
static std::map<uint64_t, std::map<uint32_t, Write>> g_setwrites;   // set -> binding*1000+element -> write
static std::map<uint64_t, std::string> g_names;
static int g_next = 1, g_dispatch = 0, g_upload = 0;
static bool g_tracing = false;
static VkDevice g_device;

template <class T> static uint64_t H(T h) { return (uint64_t)h; }
static std::string hex(const void* p, size_t n) {
    std::string s;
    char b[4];
    for (size_t i = 0; i < n; i++) snprintf(b, sizeof b, "%02x", ((const uint8_t*)p)[i]), s += b;
    return s;
}
static void dump_file(const std::string& name, const void* p, size_t n) {
    FILE* f = fopen((g_out + "/" + name).c_str(), "wb");
    if (!f) die("cannot write %s", name.c_str());
    fwrite(p, 1, n, f);
    fclose(f);
}
static const void* buffer_bytes(uint64_t buffer, VkDeviceSize offset) {
    auto b = g_buffers.find(buffer);
    if (b == g_buffers.end()) return nullptr;
    auto m = g_mapped.find(b->second.memory);
    if (m == g_mapped.end() || !m->second) return nullptr;
    return (const uint8_t*)m->second + b->second.offset + offset;
}

// ---------------------------------------------------------------- the hooks
namespace hook {
VkResult VKAPI_CALL CreateBuffer(VkDevice d, const VkBufferCreateInfo* ci, const VkAllocationCallbacks* a, VkBuffer* out) {
    VkBufferCreateInfo c = *ci;
    c.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkResult r = vkCreateBuffer(d, &c, a, out);
    if (r == VK_SUCCESS) {
        g_buffers[H(*out)] = {g_next, ci->size, ci->usage};
        logf("CreateBuffer buf%d size=%llu usage=0x%x", g_next++, (unsigned long long)ci->size, ci->usage);
    } else logf("CreateBuffer FAILED %d size=%llu", r, (unsigned long long)ci->size);
    return r;
}
VkResult VKAPI_CALL CreateImage(VkDevice d, const VkImageCreateInfo* ci, const VkAllocationCallbacks* a, VkImage* out) {
    VkImageCreateInfo c = *ci;
    c.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (c.format == VK_FORMAT_R64_UINT) c.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;   // (sampled as R32G32_UINT: tools/spv_fix.py)
    VkResult r = vkCreateImage(d, &c, a, out);
    if (r == VK_SUCCESS) {
        g_images[H(*out)] = {g_next, ci->format, ci->extent, ci->mipLevels, ci->usage, ci->initialLayout};
        logf("CreateImage img%d type=%d format=%d extent=%ux%ux%u mips=%u layers=%u usage=0x%x flags=0x%x", g_next++,
             ci->imageType, ci->format, ci->extent.width, ci->extent.height, ci->extent.depth, ci->mipLevels,
             ci->arrayLayers, ci->usage, ci->flags);
    } else logf("CreateImage FAILED %d format=%d", r, ci->format);
    return r;
}
VkResult VKAPI_CALL CreateImageView(VkDevice d, const VkImageViewCreateInfo* ci, const VkAllocationCallbacks* a, VkImageView* out) {
    VkResult r = vkCreateImageView(d, ci, a, out);
    if (r == VK_SUCCESS) {
        g_views[H(*out)] = {g_next, H(ci->image), ci->subresourceRange.baseMipLevel, *ci};
        auto im = g_images.find(H(ci->image));
        logf("CreateImageView view%d img%d format=%d viewType=%d mip=%u+%u aspect=0x%x", g_next++,
             im == g_images.end() ? -1 : im->second.id, ci->format, ci->viewType, ci->subresourceRange.baseMipLevel,
             ci->subresourceRange.levelCount, ci->subresourceRange.aspectMask);
    }
    return r;
}
VkResult VKAPI_CALL CreateSampler(VkDevice d, const VkSamplerCreateInfo* ci, const VkAllocationCallbacks* a, VkSampler* out) {
    VkResult r = vkCreateSampler(d, ci, a, out);
    if (r == VK_SUCCESS && !g_samplers.count(H(*out))) {
        g_samplers[H(*out)] = g_next;
        logf("CreateSampler smp%d mag=%d min=%d mip=%d addr=%d,%d,%d maxLod=%g", g_next++, ci->magFilter, ci->minFilter,
             ci->mipmapMode, ci->addressModeU, ci->addressModeV, ci->addressModeW, ci->maxLod);
    }
    return r;
}
VkResult VKAPI_CALL CreateShaderModule(VkDevice d, const VkShaderModuleCreateInfo* ci, const VkAllocationCallbacks* a, VkShaderModule* out) {
    char name[64];
    snprintf(name, sizeof name, "mod_%04d.spv", g_next);
    dump_file(name, ci->pCode, ci->codeSize);
    // (a rewritten module in place of AMD's, by the FNV-1a of AMD's: tools/spv_fix.py, for what Mesa does not take)
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < ci->codeSize; i++) h = (h ^ ((const uint8_t*)ci->pCode)[i]) * 0x100000001b3ull;
    VkShaderModuleCreateInfo c = *ci;
    std::vector<uint32_t> patched;
    if (const char* dir = getenv("TRACE_PATCHES")) {
        char path[600];
        snprintf(path, sizeof path, "%s/%016llx.spv", dir, (unsigned long long)h);
        if (FILE* f = fopen(path, "rb")) {
            fseek(f, 0, SEEK_END);
            patched.resize(ftell(f) / 4);
            fseek(f, 0, SEEK_SET);
            fread(patched.data(), 4, patched.size(), f);
            fclose(f);
            c.pCode = patched.data(), c.codeSize = patched.size() * 4;
        }
    }
    VkResult r = vkCreateShaderModule(d, &c, a, out);
    logf("CreateShaderModule mod%d size=%llu fnv=%016llx%s result=%d", g_next, (unsigned long long)ci->codeSize, (unsigned long long)h,
         patched.empty() ? "" : " PATCHED", r);
    if (r == VK_SUCCESS) g_modules[H(*out)] = g_next;
    g_next++;
    return r;
}
VkResult VKAPI_CALL CreateDescriptorSetLayout(VkDevice d, const VkDescriptorSetLayoutCreateInfo* ci, const VkAllocationCallbacks* a, VkDescriptorSetLayout* out) {
    VkResult r = vkCreateDescriptorSetLayout(d, ci, a, out);
    std::string s;
    char b[64];
    for (uint32_t i = 0; i < ci->bindingCount; i++) {
        snprintf(b, sizeof b, " %u:t%d:n%u", ci->pBindings[i].binding, ci->pBindings[i].descriptorType, ci->pBindings[i].descriptorCount);
        s += b;
    }
    if (r == VK_SUCCESS) g_setlayouts[H(*out)] = g_next;
    logf("CreateDescriptorSetLayout dsl%d flags=0x%x result=%d bindings:%s", g_next++, ci->flags, r, s.c_str());
    return r;
}
VkResult VKAPI_CALL CreatePipelineLayout(VkDevice d, const VkPipelineLayoutCreateInfo* ci, const VkAllocationCallbacks* a, VkPipelineLayout* out) {
    VkResult r = vkCreatePipelineLayout(d, ci, a, out);
    std::string s;
    char b[64];
    for (uint32_t i = 0; i < ci->setLayoutCount; i++) snprintf(b, sizeof b, " dsl%d", g_setlayouts[H(ci->pSetLayouts[i])]), s += b;
    for (uint32_t i = 0; i < ci->pushConstantRangeCount; i++)
        snprintf(b, sizeof b, " push(%u+%u)", ci->pPushConstantRanges[i].offset, ci->pPushConstantRanges[i].size), s += b;
    if (r == VK_SUCCESS) g_pipelayouts[H(*out)] = g_next;
    logf("CreatePipelineLayout pl%d result=%d:%s", g_next++, r, s.c_str());
    return r;
}
VkResult VKAPI_CALL CreateComputePipelines(VkDevice d, VkPipelineCache cache, uint32_t n, const VkComputePipelineCreateInfo* ci, const VkAllocationCallbacks* a, VkPipeline* out) {
    VkResult r = vkCreateComputePipelines(d, cache, n, ci, a, out);
    for (uint32_t i = 0; i < n; i++) {
        std::string chain;
        char b[64];
        for (auto* p = (const VkBaseInStructure*)ci[i].stage.pNext; p; p = p->pNext) {
            snprintf(b, sizeof b, " stageNext=%d", p->sType), chain += b;
            if (p->sType == VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO)
                snprintf(b, sizeof b, "(subgroup=%u)", ((const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo*)p)->requiredSubgroupSize), chain += b;
        }
        if (ci[i].stage.pSpecializationInfo) chain += " SPECIALIZED";
        int mod = g_modules[H(ci[i].stage.module)];
        if (r == VK_SUCCESS) g_pipes[H(out[i])] = {g_next, mod, ci[i].stage.pName, H(ci[i].layout)};
        logf("CreateComputePipeline pipe%d mod%d entry=%s pl%d flags=0x%x stageFlags=0x%x result=%d%s", g_next++, mod,
             ci[i].stage.pName, g_pipelayouts[H(ci[i].layout)], ci[i].flags, ci[i].stage.flags, r, chain.c_str());
    }
    return r;
}
VkResult VKAPI_CALL AllocateMemory(VkDevice d, const VkMemoryAllocateInfo* ai, const VkAllocationCallbacks* a, VkDeviceMemory* out) {
    VkResult r = vkAllocateMemory(d, ai, a, out);
    if (r == VK_SUCCESS) g_memories[H(*out)] = g_next;
    logf("AllocateMemory mem%d size=%llu type=%u result=%d", g_next++, (unsigned long long)ai->allocationSize, ai->memoryTypeIndex, r);
    return r;
}
VkResult VKAPI_CALL MapMemory(VkDevice d, VkDeviceMemory m, VkDeviceSize off, VkDeviceSize size, VkMemoryMapFlags f, void** out) {
    VkResult r = vkMapMemory(d, m, off, size, f, out);
    if (r == VK_SUCCESS) g_mapped[H(m)] = (uint8_t*)*out - off;
    return r;
}
void VKAPI_CALL UnmapMemory(VkDevice d, VkDeviceMemory m) {
    // (kept mapped for the trace: an upload's bytes are read when its copy is recorded, which may be after the unmap)
    (void)d, (void)m;
}
VkResult VKAPI_CALL BindBufferMemory(VkDevice d, VkBuffer b, VkDeviceMemory m, VkDeviceSize off) {
    auto it = g_buffers.find(H(b));
    if (it != g_buffers.end()) it->second.memory = H(m), it->second.offset = off;
    return vkBindBufferMemory(d, b, m, off);
}
VkResult VKAPI_CALL AllocateDescriptorSets(VkDevice d, const VkDescriptorSetAllocateInfo* ai, VkDescriptorSet* out) {
    VkResult r = vkAllocateDescriptorSets(d, ai, out);
    if (r == VK_SUCCESS)
        for (uint32_t i = 0; i < ai->descriptorSetCount; i++) {
            g_sets[H(out[i])] = g_next;
            g_setwrites[H(out[i])].clear();
            logf("AllocateDescriptorSet set%d dsl%d", g_next++, g_setlayouts[H(ai->pSetLayouts[i])]);
        }
    else logf("AllocateDescriptorSets FAILED %d", r);
    return r;
}
void VKAPI_CALL UpdateDescriptorSets(VkDevice d, uint32_t n, const VkWriteDescriptorSet* w, uint32_t nc, const VkCopyDescriptorSet* c) {
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t k = 0; k < w[i].descriptorCount; k++) {
            Write x{w[i].descriptorType};
            switch (w[i].descriptorType) {
            case VK_DESCRIPTOR_TYPE_SAMPLER: x.b = H(w[i].pImageInfo[k].sampler); break;
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                x.a = H(w[i].pImageInfo[k].imageView), x.b = H(w[i].pImageInfo[k].sampler), x.layout = w[i].pImageInfo[k].imageLayout;
                break;
            default:
                if (w[i].pBufferInfo) x.a = H(w[i].pBufferInfo[k].buffer), x.offset = w[i].pBufferInfo[k].offset, x.range = w[i].pBufferInfo[k].range;
            }
            g_setwrites[H(w[i].dstSet)][w[i].dstBinding * 1000 + w[i].dstArrayElement + k] = x;
        }
    if (nc) logf("UpdateDescriptorSets with %u COPIES (not traced)", nc);
    // (a sampled 64-bit image is bound as its two 32-bit halves: tools/spv_fix.py)
    std::vector<VkWriteDescriptorSet> ws(w, w + n);
    std::vector<std::vector<VkDescriptorImageInfo>> swapped;
    swapped.reserve(n);
    for (auto& x : ws) {
        if (x.descriptorType != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) continue;
        bool any = false;
        std::vector<VkDescriptorImageInfo> infos(x.pImageInfo, x.pImageInfo + x.descriptorCount);
        for (auto& info : infos) {
            auto v = g_views.find(H(info.imageView));
            if (v == g_views.end()) continue;
            auto im = g_images.find(v->second.image);
            if (im == g_images.end() || im->second.format != VK_FORMAT_R64_UINT) continue;
            if (!v->second.as_rg32) {
                VkImageViewCreateInfo vc = v->second.ci;
                vc.pNext = nullptr, vc.format = VK_FORMAT_R32G32_UINT;
                if (vkCreateImageView(d, &vc, nullptr, &v->second.as_rg32)) logf("R32G32 view FAILED");
            }
            info.imageView = v->second.as_rg32, any = true;
        }
        if (any) swapped.push_back(std::move(infos)), x.pImageInfo = swapped.back().data();
    }
    vkUpdateDescriptorSets(d, n, ws.data(), nc, c);
}

struct Bound {
    uint64_t pipeline = 0;
    std::vector<uint64_t> sets;
    std::vector<uint32_t> dynamic;
};
static std::map<uint64_t, Bound> g_bound;   // by command buffer

void VKAPI_CALL CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p) {
    g_bound[H(cb)].pipeline = H(p);
    vkCmdBindPipeline(cb, bp, p);
}
void VKAPI_CALL CmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first, uint32_t n, const VkDescriptorSet* sets, uint32_t nd, const uint32_t* dyn) {
    Bound& b = g_bound[H(cb)];
    if (b.sets.size() < first + n) b.sets.resize(first + n);
    for (uint32_t i = 0; i < n; i++) b.sets[first + i] = H(sets[i]);
    b.dynamic.assign(dyn, dyn + nd);
    vkCmdBindDescriptorSets(cb, bp, l, first, n, sets, nd, dyn);
}
static void describe_bound(VkCommandBuffer cb, const char* what, uint32_t x, uint32_t y, uint32_t z) {
    Bound& b = g_bound[H(cb)];
    auto p = g_pipes.find(b.pipeline);
    logf("DISPATCH %d %s %ux%ux%u pipe%d mod%d entry=%s", g_dispatch, what, x, y, z, p == g_pipes.end() ? -1 : p->second.id,
         p == g_pipes.end() ? -1 : p->second.module, p == g_pipes.end() ? "?" : p->second.entry.c_str());
    size_t dyn = 0;
    for (size_t s = 0; s < b.sets.size(); s++) {
        for (auto& [key, w] : g_setwrites[b.sets[s]]) {
            const uint32_t binding = key / 1000, element = key % 1000;
            switch (w.type) {
            case VK_DESCRIPTOR_TYPE_SAMPLER: logf("  s%zu b%u[%u] sampler smp%d", s, binding, element, g_samplers[w.b]); break;
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
                auto v = g_views.find(w.a);
                if (v == g_views.end()) {
                    logf("  s%zu b%u[%u] %s view? layout=%d", s, binding, element, w.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? "storage" : "sampled", w.layout);
                    break;
                }
                auto im = g_images.find(v->second.image);
                logf("  s%zu b%u[%u] %s img%d mip%u layout=%d %s", s, binding, element,
                     w.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? "storage" : "sampled", im == g_images.end() ? -1 : im->second.id,
                     v->second.mip, w.layout, im == g_images.end() ? "" : im->second.name.c_str());
                break;
            }
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: {
                VkDeviceSize off = w.offset;
                if (w.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC && dyn < b.dynamic.size()) off += b.dynamic[dyn++];
                auto bi = g_buffers.find(w.a);
                const void* bytes = buffer_bytes(w.a, off);
                const VkDeviceSize range = w.range == VK_WHOLE_SIZE ? (bi == g_buffers.end() ? 0 : bi->second.size - off) : w.range;
                logf("  s%zu b%u[%u] ubo buf%d off=%llu range=%llu data=%s", s, binding, element, bi == g_buffers.end() ? -1 : bi->second.id,
                     (unsigned long long)off, (unsigned long long)range, bytes ? hex(bytes, (size_t)std::min<VkDeviceSize>(range, 8192)).c_str() : "(unmapped)");
                break;
            }
            default: {
                auto bi = g_buffers.find(w.a);
                logf("  s%zu b%u[%u] buffer(t%d) buf%d off=%llu range=%lld %s", s, binding, element, w.type, bi == g_buffers.end() ? -1 : bi->second.id,
                     (unsigned long long)w.offset, (long long)w.range, bi == g_buffers.end() ? "" : bi->second.name.c_str());
            }
            }
        }
    }
    g_dispatch++;
}
static bool g_stopped = false;
void VKAPI_CALL CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    // TRACE_STOP_ENTRY=<entry>: every dispatch after that pass is left out (to compare the scratch buffer at that point)
    if (g_stopped) return;
    describe_bound(cb, "direct", x, y, z);
    vkCmdDispatch(cb, x, y, z);
    static const char* stop = getenv("TRACE_STOP_ENTRY");
    auto p = g_pipes.find(g_bound[H(cb)].pipeline);
    if (stop && p != g_pipes.end() && p->second.entry == stop) g_stopped = true;
}
void VKAPI_CALL CmdDispatchIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off) {
    describe_bound(cb, "INDIRECT", 0, 0, 0);
    vkCmdDispatchIndirect(cb, b, off);
}
void VKAPI_CALL CmdPushConstants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st, uint32_t off, uint32_t size, const void* v) {
    logf("PushConstants off=%u size=%u data=%s", off, size, hex(v, size).c_str());
    vkCmdPushConstants(cb, l, st, off, size, v);
}
void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer cb, VkBuffer src, VkBuffer dst, uint32_t n, const VkBufferCopy* r) {
    for (uint32_t i = 0; i < n; i++) {
        const void* bytes = buffer_bytes(H(src), r[i].srcOffset);
        char name[64] = "-";
        if (bytes) snprintf(name, sizeof name, "upload_%04d.bin", g_upload++), dump_file(name, bytes, (size_t)r[i].size);
        logf("CopyBuffer buf%d+%llu -> buf%d+%llu size=%llu file=%s", g_buffers[H(src)].id, (unsigned long long)r[i].srcOffset,
             g_buffers[H(dst)].id, (unsigned long long)r[i].dstOffset, (unsigned long long)r[i].size, name);
    }
    vkCmdCopyBuffer(cb, src, dst, n, r);
}
void VKAPI_CALL CmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer src, VkImage dst, VkImageLayout l, uint32_t n, const VkBufferImageCopy* r) {
    auto im = g_images.find(H(dst));
    for (uint32_t i = 0; i < n; i++) {
        const void* bytes = buffer_bytes(H(src), r[i].bufferOffset);
        char name[64] = "-";
        // (the size of the copy is not known without the format's texel size: the rest of the staging buffer is written)
        auto bi = g_buffers.find(H(src));
        if (bytes && bi != g_buffers.end())
            snprintf(name, sizeof name, "upload_%04d.bin", g_upload++), dump_file(name, bytes, (size_t)(bi->second.size - r[i].bufferOffset));
        logf("CopyBufferToImage buf%d+%llu -> img%d mip%u extent=%ux%ux%u file=%s", g_buffers[H(src)].id, (unsigned long long)r[i].bufferOffset,
             im == g_images.end() ? -1 : im->second.id, r[i].imageSubresource.mipLevel, r[i].imageExtent.width, r[i].imageExtent.height,
             r[i].imageExtent.depth, name);
    }
    vkCmdCopyBufferToImage(cb, src, dst, l, n, r);
}
void VKAPI_CALL CmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst, VkImageLayout dl, uint32_t n, const VkImageCopy* r) {
    if (g_stopped) return;
    for (uint32_t i = 0; i < n; i++)
        logf("CopyImage img%d mip%u (%d,%d) -> img%d mip%u (%d,%d) extent=%ux%u", g_images[H(src)].id, r[i].srcSubresource.mipLevel, r[i].srcOffset.x,
             r[i].srcOffset.y, g_images[H(dst)].id, r[i].dstSubresource.mipLevel, r[i].dstOffset.x, r[i].dstOffset.y, r[i].extent.width, r[i].extent.height);
    vkCmdCopyImage(cb, src, sl, dst, dl, n, r);
}
void VKAPI_CALL CmdClearColorImage(VkCommandBuffer cb, VkImage im, VkImageLayout l, const VkClearColorValue* c, uint32_t n, const VkImageSubresourceRange* r) {
    logf("ClearColorImage img%d color=%g,%g,%g,%g (u32 %u,%u,%u,%u) mips=%u+%u", g_images[H(im)].id, c->float32[0], c->float32[1], c->float32[2],
         c->float32[3], c->uint32[0], c->uint32[1], c->uint32[2], c->uint32[3], n ? r[0].baseMipLevel : 0, n ? r[0].levelCount : 0);
    vkCmdClearColorImage(cb, im, l, c, n, r);
}
void VKAPI_CALL CmdFillBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off, VkDeviceSize size, uint32_t data) {
    logf("FillBuffer buf%d off=%llu size=%lld data=0x%x", g_buffers[H(b)].id, (unsigned long long)off, (long long)size, data);
    vkCmdFillBuffer(cb, b, off, size, data);
}
void VKAPI_CALL CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags ss, VkPipelineStageFlags ds, VkDependencyFlags df, uint32_t nm, const VkMemoryBarrier* m, uint32_t nb, const VkBufferMemoryBarrier* b, uint32_t ni, const VkImageMemoryBarrier* im) {
    for (uint32_t i = 0; i < ni; i++) {
        auto it = g_images.find(H(im[i].image));
        if (it != g_images.end()) it->second.layout = im[i].newLayout;
    }
    vkCmdPipelineBarrier(cb, ss, ds, df, nm, m, nb, b, ni, im);
}
VkResult VKAPI_CALL SetDebugUtilsObjectNameEXT(VkDevice, const VkDebugUtilsObjectNameInfoEXT* info) {
    const char* n = info->pObjectName ? info->pObjectName : "";
    g_names[info->objectHandle] = n;
    if (auto i = g_images.find(info->objectHandle); i != g_images.end()) i->second.name = n, logf("Name img%d = %s", i->second.id, n);
    else if (auto b = g_buffers.find(info->objectHandle); b != g_buffers.end()) b->second.name = n, logf("Name buf%d = %s", b->second.id, n);
    else if (auto p = g_pipes.find(info->objectHandle); p != g_pipes.end()) logf("Name pipe%d = %s", p->second.id, n);
    else logf("Name type%d %llx = %s", info->objectType, (unsigned long long)info->objectHandle, n);
    return VK_SUCCESS;
}
void VKAPI_CALL CmdBeginDebugUtilsLabelEXT(VkCommandBuffer, const VkDebugUtilsLabelEXT* l) { logf("LABEL begin %s", l->pLabelName); }
void VKAPI_CALL CmdEndDebugUtilsLabelEXT(VkCommandBuffer) { logf("LABEL end"); }

PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice d, const char* name) {
#define HOOK(n) \
    if (!strcmp(name, "vk" #n)) return (PFN_vkVoidFunction)n;
    HOOK(CreateBuffer) HOOK(CreateImage) HOOK(CreateImageView) HOOK(CreateSampler) HOOK(CreateShaderModule)
    HOOK(CreateDescriptorSetLayout) HOOK(CreatePipelineLayout) HOOK(CreateComputePipelines) HOOK(AllocateMemory)
    HOOK(MapMemory) HOOK(UnmapMemory) HOOK(BindBufferMemory) HOOK(AllocateDescriptorSets) HOOK(UpdateDescriptorSets)
    HOOK(CmdBindPipeline) HOOK(CmdBindDescriptorSets) HOOK(CmdDispatch) HOOK(CmdDispatchIndirect) HOOK(CmdPushConstants)
    HOOK(CmdCopyBuffer) HOOK(CmdCopyBufferToImage) HOOK(CmdCopyImage) HOOK(CmdClearColorImage) HOOK(CmdFillBuffer)
    HOOK(CmdPipelineBarrier) HOOK(SetDebugUtilsObjectNameEXT) HOOK(CmdBeginDebugUtilsLabelEXT) HOOK(CmdEndDebugUtilsLabelEXT)
#undef HOOK
    PFN_vkVoidFunction f = vkGetDeviceProcAddr(d, name);
    logf("GetDeviceProcAddr %s -> %s (not traced)", name, f ? "real" : "NULL");
    return f;
}
} // namespace hook

// ---------------------------------------------------------------- our own images and the scene
static VkPhysicalDevice g_phys;
static VkQueue g_queue;
static VkCommandBuffer g_cmd;
static VkPhysicalDeviceMemoryProperties g_memprops;

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g_memprops.memoryTypeCount; i++)
        if ((bits >> i & 1) && (g_memprops.memoryTypes[i].propertyFlags & want) == want) return i;
    die("no memory type");
}
struct Staging {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void* map;
    VkDeviceSize size;
};
static Staging staging(VkDeviceSize size) {
    Staging s{};
    s.size = size;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size, ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(g_device, &ci, nullptr, &s.buffer)) die("staging buffer");
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_device, s.buffer, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(g_device, &ai, nullptr, &s.memory)) die("staging memory");
    vkBindBufferMemory(g_device, s.buffer, s.memory, 0);
    vkMapMemory(g_device, s.memory, 0, VK_WHOLE_SIZE, 0, &s.map);
    return s;
}
static void free_staging(Staging& s) {
    vkDestroyBuffer(g_device, s.buffer, nullptr);
    vkFreeMemory(g_device, s.memory, nullptr);
}
static VkImage our_image(VkFormat format, uint32_t w, uint32_t h, const char* name) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D, ci.format = format, ci.extent = {w, h, 1}, ci.mipLevels = 1, ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImage im;
    if (vkCreateImage(g_device, &ci, nullptr, &im)) die("image %s", name);
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(g_device, im, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size, ai.memoryTypeIndex = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory mem;
    if (vkAllocateMemory(g_device, &ai, nullptr, &mem)) die("image memory");
    vkBindImageMemory(g_device, im, mem, 0);
    ImageInfo info{g_next, format, ci.extent, 1, ci.usage, VK_IMAGE_LAYOUT_UNDEFINED, name, true};
    g_images[H(im)] = info;
    logf("OurImage img%d %s format=%d %ux%u", g_next++, name, format, w, h);
    return im;
}
static void begin() {
    vkResetCommandBuffer(g_cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(g_cmd, &bi);
}
static void submit() {
    vkEndCommandBuffer(g_cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1, si.pCommandBuffers = &g_cmd;
    if (VkResult r = vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE)) die("submit %d", r);
    if (VkResult r = vkQueueWaitIdle(g_queue)) die("wait idle %d (device lost?)", r);
}
static void to_layout(VkImage im, VkImageLayout to, uint32_t mips = 1) {
    ImageInfo& info = g_images[H(im)];
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = info.layout, b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im, b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
    vkCmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    info.layout = to;
}
static uint32_t texel_bytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_SINT: return 1;
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16_SNORM: return 2;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_R8G8B8A8_SNORM: case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT:
    case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R64_UINT: case VK_FORMAT_R64_SINT: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: return 16;
    default: return 0;
    }
}
static void upload(VkImage im, const void* data, size_t bytes) {
    ImageInfo& info = g_images[H(im)];
    Staging s = staging(bytes);
    memcpy(s.map, data, bytes);
    begin();
    to_layout(im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, r.imageExtent = info.extent;
    vkCmdCopyBufferToImage(g_cmd, s.buffer, im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    to_layout(im, VK_IMAGE_LAYOUT_GENERAL);
    submit();
    free_staging(s);
}
/// Every image the provider made (and ours), each mip, as raw texels: <prefix>_img<id>_m<mip>.raw
static void dump_images(const char* prefix, bool ours_only) {
    for (auto& [handle, info] : g_images) {
        const uint32_t tb = texel_bytes(info.format);
        if (ours_only != info.ours && ours_only) continue;
        if (!tb || info.layout == VK_IMAGE_LAYOUT_UNDEFINED || info.extent.depth != 1) {
            if (!tb) logf("dump: img%d format %d not dumped", info.id, info.format);
            continue;
        }
        for (uint32_t mip = 0; mip < info.mips; mip++) {
            const uint32_t w = std::max(1u, info.extent.width >> mip), h = std::max(1u, info.extent.height >> mip);
            Staging s = staging((VkDeviceSize)w * h * tb);
            const VkImageLayout was = info.layout;
            begin();
            to_layout((VkImage)handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, info.mips);
            VkBufferImageCopy r{};
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1}, r.imageExtent = {w, h, 1};
            vkCmdCopyImageToBuffer(g_cmd, (VkImage)handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.buffer, 1, &r);
            to_layout((VkImage)handle, was, info.mips);
            submit();
            char name[128];
            snprintf(name, sizeof name, "%s_img%d_m%u.raw", prefix, info.id, mip);
            dump_file(name, s.map, (size_t)s.size);
            logf("dump %s img%d mip%u %ux%u format=%d name=%s", name, info.id, mip, w, h, info.format, info.name.c_str());
            free_staging(s);
        }
    }
}
static void dump_buffers(const char* prefix, VkDeviceSize limit) {
    for (auto& [handle, info] : g_buffers) {
        if (!(info.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) || info.size > limit) continue;
        Staging s = staging(info.size);
        begin();
        VkBufferCopy r{0, 0, info.size};
        vkCmdCopyBuffer(g_cmd, (VkBuffer)handle, s.buffer, 1, &r);
        submit();
        char name[128];
        snprintf(name, sizeof name, "%s_buf%d.raw", prefix, info.id);
        dump_file(name, s.map, (size_t)s.size);
        logf("dump %s buf%d size=%llu name=%s", name, info.id, (unsigned long long)info.size, info.name.c_str());
        free_staging(s);
    }
}

static uint16_t half(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int exp = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man = (man | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | ((man + 0x1000) >> 13));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (exp << 10) | ((man + 0x1000) >> 13));
}
/// sin and cos from + - * / and floor only, so the mingw and glibc builds draw the same pictures and send the same camera
/// (their libm sin() differs in the last bit for some inputs). Accuracy is beside the point; determinism is the point.
static double det_sin(double x) {
    const double two_pi = 6.283185307179586, pi = 3.141592653589793;
    x -= two_pi * std::floor(x / two_pi + 0.5);   // [-pi, pi]
    if (x > pi * 0.5) x = pi - x;
    if (x < -pi * 0.5) x = -pi - x;               // [-pi/2, pi/2]
    const double x2 = x * x;   // (Taylor to x^15: under 1e-10 off on this range)
    double p = -1.0 / 1307674368000.0;
    p = p * x2 + 1.0 / 6227020800.0;
    p = p * x2 - 1.0 / 39916800.0;
    p = p * x2 + 1.0 / 362880.0;
    p = p * x2 - 1.0 / 5040.0;
    p = p * x2 + 1.0 / 120.0;
    p = p * x2 - 1.0 / 6.0;
    p = p * x2 + 1.0;
    return x * p;
}
static float dsin(float x) { return (float)det_sin(x); }
static float dcos(float x) { return (float)det_sin((double)x + 1.5707963267948966); }
static uint32_t hash2(int x, int y) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}
/// The camera of a frame: still, or (TRACE_CAMERA=k) moving and turning a little more each frame, k times as fast
struct Cam { float pos[3], right[3], up[3], fwd[3]; };
static Cam camera_at(int frame) {
    const float t = getenv("TRACE_CAMERA") ? (float)frame * (float)atof(getenv("TRACE_CAMERA")) : 0.0f;
    const float yaw = 0.07f * t + 0.013f * t * t, pitch = 0.03f * t;
    Cam c;
    const float f[3] = {dsin(yaw) * dcos(pitch), dsin(pitch), dcos(yaw) * dcos(pitch)};
    const float rgt[3] = {dcos(yaw), 0.0f, -dsin(yaw)};
    const float up[3] = {f[1] * rgt[2] - f[2] * rgt[1], f[2] * rgt[0] - f[0] * rgt[2], f[0] * rgt[1] - f[1] * rgt[0]};
    const float pos[3] = {0.31f * t, 0.11f * t * t, -0.23f * t};
    for (int i = 0; i < 3; i++) c.pos[i] = pos[i], c.right[i] = rgt[i], c.up[i] = up[i], c.fwd[i] = f[i];
    return c;
}
static double dot3(const double* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

/// A background wall seen through the camera (its motion is the camera's, so AMD's forward motion pass takes its
/// camera path there) and a nearer square that moves across the screen on its own (so it takes the splatted flow)., with the depth and motion that go with them.
/// Motion is in pixels at render size, pointing from this frame's pixel to where it was (FSR's convention).
/// The camera: x right, y up, looking along +z, vertical field of view fov, aspect of the render size.
struct Scene {
    uint32_t w, h, rw, rh;
    bool inverted, infinite;   // (the depth convention of the create flags)
    float near_, far_, fov;
    uint32_t mw, mh;   // (the motion vectors' size: the render size, or the display size with DISPLAY_RESOLUTION_MOTION_VECTORS)
    static constexpr double WALL_Z = 20.0;   // (the wall: the plane z = 20)
    /// depth at a view distance d, by the create flags' convention
    float depth_at(float d) const {
        const double n = near_, f = far_, z = infinite ? 1.0 - n / d : f * (d - n) / (d * (f - n));
        return (float)(inverted ? 1.0 - z : z);
    }
    std::vector<uint8_t> color;
    std::vector<float> depth;
    std::vector<uint16_t> motion;
    static void background(float x, float y, uint8_t* out) {
        const int cx = (int)std::floor(x / 48.0f), cy = (int)std::floor(y / 48.0f);
        const uint32_t hsh = hash2(cx, cy);
        const float fx = x / 48.0f - cx, fy = y / 48.0f - cy;
        const float edge = (fx < 0.06f || fy < 0.06f) ? 0.35f : 1.0f;
        const float stripe = 0.75f + 0.25f * dsin((x + y) * 0.21f);
        out[0] = (uint8_t)(edge * stripe * (60 + (hsh & 0x7f)));
        out[1] = (uint8_t)(edge * stripe * (60 + ((hsh >> 8) & 0x7f)));
        out[2] = (uint8_t)(edge * stripe * (60 + ((hsh >> 16) & 0x7f)));
        out[3] = 255;
    }
    /// where the ray through (u, v) of camera c meets the wall: the point, and its distance along c's forward
    void wall(const Cam& c, double u, double v, double aspect, double* p, double* dist) const {
        const double th = det_sin(fov * 0.5) / det_sin(fov * 0.5 + 1.5707963267948966);
        const double nx = (2.0 * u - 1.0) * th * aspect, ny = (1.0 - 2.0 * v) * th;
        double d[3];
        for (int i = 0; i < 3; i++) d[i] = c.fwd[i] + c.right[i] * nx + c.up[i] * ny;
        const double t = (WALL_Z - c.pos[2]) / d[2];   // (d . fwd = 1, so t is also the view depth)
        for (int i = 0; i < 3; i++) p[i] = c.pos[i] + t * d[i];
        *dist = t;
    }
    /// where world point p is in camera c's picture, as (u, v)
    void project(const Cam& c, const double* p, double aspect, double* u, double* v) const {
        const double th = det_sin(fov * 0.5) / det_sin(fov * 0.5 + 1.5707963267948966);
        const double q[3] = {p[0] - c.pos[0], p[1] - c.pos[1], p[2] - c.pos[2]};
        const double z = dot3(q, c.fwd);
        *u = (dot3(q, c.right) / (z * th * aspect) + 1.0) * 0.5;
        *v = (1.0 - dot3(q, c.up) / (z * th)) * 0.5;
    }
    void make(int frame) {
        color.resize((size_t)w * h * 4), depth.resize((size_t)rw * rh), motion.resize((size_t)mw * mh * 2);
        const Cam cam = camera_at(frame), prev = camera_at(frame > 0 ? frame - 1 : 0);
        // (the square: fractions of the picture a frame, so the same scene at any size)
        const float sqx = 0.70f - 0.017f * frame, sqy = 0.35f + 0.006f * frame, sqvx = -0.017f, sqvy = 0.006f, sqr = 0.11f;
        const double aspect = (double)rw / rh;
        for (uint32_t y = 0; y < h; y++)
            for (uint32_t x = 0; x < w; x++) {
                const float u = (x + 0.5f) / w, v = (y + 0.5f) / h;
                uint8_t* px = &color[((size_t)y * w + x) * 4];
                if (std::fabs(u - sqx) < sqr * h / w && std::fabs(v - sqy) < sqr) {
                    const float lu = (u - sqx) * w / h, lv = v - sqy;
                    const bool check = (((int)std::floor(lu * 40) + (int)std::floor(lv * 40)) & 1) != 0;
                    px[0] = check ? 230 : 40, px[1] = check ? 200 : 90, px[2] = check ? 40 : 200, px[3] = 255;
                } else {
                    double p[3], dist;
                    wall(cam, u, v, aspect, p, &dist);
                    background((float)(p[0] * 50.0), (float)(-p[1] * 50.0), px);   // (about a cell of 48 pixels per unit)
                }
            }
        // (depth at the render size, motion at its own; each the same scene)
        auto sample = [&](uint32_t x, uint32_t y, uint32_t sw, uint32_t sh, float* d, float* mx, float* my) {
            const float u = (x + 0.5f) / sw, v = (y + 0.5f) / sh;
            if (std::fabs(u - sqx) < sqr * sh / sw && std::fabs(v - sqy) < sqr) {
                *d = 2.0f, *mx = -sqvx * sw, *my = -sqvy * sh;   // (the square 2 units away)
            } else {
                double p[3], dist, pu, pv;
                wall(cam, u, v, aspect, p, &dist);
                project(prev, p, aspect, &pu, &pv);
                *d = (float)dist, *mx = (float)((pu - u) * sw), *my = (float)((pv - v) * sh);
            }
        };
        for (uint32_t y = 0; y < rh; y++)
            for (uint32_t x = 0; x < rw; x++) {
                float d, mx, my;
                sample(x, y, rw, rh, &d, &mx, &my);
                depth[(size_t)y * rw + x] = depth_at(d);
            }
        for (uint32_t y = 0; y < mh; y++)
            for (uint32_t x = 0; x < mw; x++) {
                float d, mx, my;
                sample(x, y, mw, mh, &d, &mx, &my);
                motion[((size_t)y * mw + x) * 2 + 0] = half(mx);
                motion[((size_t)y * mw + x) * 2 + 1] = half(my);
            }
    }
};

/// The scene's 8-bit RGBA in the back buffer's format: 8-bit formats as they are (BGRA swapped), 10-bit as v * 1023 / 255
/// (integer, exact), 16-bit float as v / 255 * TRACE_COLOR_GAIN (default 1; above 1 makes values past 1.0, for scRGB).
static std::vector<uint8_t> in_format(const std::vector<uint8_t>& rgba, VkFormat f) {
    const size_t n = rgba.size() / 4;
    std::vector<uint8_t> out;
    switch (f) {
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
        out = rgba;
        for (size_t i = 0; i < n; i++) std::swap(out[i * 4], out[i * 4 + 2]);
        return out;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: {
        out.resize(n * 4);
        for (size_t i = 0; i < n; i++) {
            const uint32_t r = rgba[i * 4] * 1023u / 255u, g = rgba[i * 4 + 1] * 1023u / 255u, b = rgba[i * 4 + 2] * 1023u / 255u;
            const uint32_t v = r | g << 10 | b << 20 | 3u << 30;
            memcpy(&out[i * 4], &v, 4);
        }
        return out;
    }
    case VK_FORMAT_R16G16B16A16_SFLOAT: {
        const float gain = getenv("TRACE_COLOR_GAIN") ? (float)atof(getenv("TRACE_COLOR_GAIN")) : 1.0f;
        out.resize(n * 8);
        for (size_t i = 0; i < n; i++)
            for (int c = 0; c < 4; c++) {
                const uint16_t h = half(c == 3 ? 1.0f : rgba[i * 4 + c] / 255.0f * gain);
                memcpy(&out[i * 8 + c * 2], &h, 2);
            }
        return out;
    }
    default: return rgba;   // (8-bit RGBA, UNORM or SRGB)
    }
}

/// A HUD over the picture (TRACE_HUD): a translucent bar, a block counter that changes every frame, and a crosshair.
/// What the HUD-less colour lacks.
static void draw_hud(std::vector<uint8_t>& rgba, uint32_t w, uint32_t h, int frame) {
    auto px = [&](uint32_t x, uint32_t y) { return &rgba[((size_t)y * w + x) * 4]; };
    for (uint32_t y = h / 20; y < h / 20 + h / 16; y++)
        for (uint32_t x = w / 24; x < w / 24 + w * 3 / 10; x++) {
            uint8_t* p = px(x, y);
            for (int c = 0; c < 3; c++) p[c] = (uint8_t)((p[c] + 3u * (c == 1 ? 250u : 220u)) / 4u);
        }
    const uint32_t cell = std::max(2u, h / 60);
    for (int bit = 0; bit < 8; bit++)
        for (uint32_t y = h - 3 * cell; y < h - 2 * cell; y++)
            for (uint32_t x = w - (12 - bit) * cell; x < w - (11 - bit) * cell; x++) {
                uint8_t* p = px(x, y);
                const bool on = ((frame * 37 + 5) >> bit) & 1;
                p[0] = on ? 255 : 10, p[1] = on ? 255 : 10, p[2] = on ? 255 : 10;
            }
    for (uint32_t k = 0; k < h / 30; k++)
        for (uint32_t t = 0; t < 2; t++) {
            uint8_t* a = px(w / 2 - h / 60 + k, h / 2 + t);
            uint8_t* b = px(w / 2 + t, h / 2 - h / 60 + k);
            a[0] = b[0] = 255, a[1] = b[1] = 40, a[2] = b[2] = 40;
        }
}

static void message(uint32_t type, const wchar_t* m) {
    std::string s;
    for (; m && *m; m++) s += (char)*m;
    logf("FFX MESSAGE (%u): %s", type, s.c_str());
    fprintf(stderr, "ffx: %s\n", s.c_str());
}

int main(int argc, char** argv) {
    if (argc < 3) die("usage: trace <provider library> <out dir> [W H RW RH frames dumpFrom]");
    g_out = argv[2];
    const uint32_t W = argc > 3 ? atoi(argv[3]) : 1920, Hh = argc > 4 ? atoi(argv[4]) : 1080;
    const uint32_t RW = argc > 5 ? atoi(argv[5]) : W, RH = argc > 6 ? atoi(argv[6]) : Hh;
    const int frames = argc > 7 ? atoi(argv[7]) : 4, dump_from = argc > 8 ? atoi(argv[8]) : 2;
#ifdef _WIN32
    _mkdir(g_out.c_str());
#else
    mkdir(g_out.c_str(), 0755);
#endif
    g_log = fopen((g_out + "/trace.log").c_str(), "w");
    if (!g_log) die("cannot write to %s", g_out.c_str());

#ifdef _WIN32
    void* vk = load_lib("vulkan-1.dll");
#else
    void* vk = load_lib("libvulkan.so.1");
#endif
    if (!vk) die("no Vulkan loader");
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)sym(vk, "vkGetInstanceProcAddr");
    vkCreateInstance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(nullptr, "vkCreateInstance");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    if (VkResult r = vkCreateInstance(&ici, nullptr, &inst)) die("instance %d", r);
#define LOADI(n) n = (PFN_##n)vkGetInstanceProcAddr(inst, #n);
    INSTANCE_FUNCS(LOADI)
    uint32_t n = 8;
    VkPhysicalDevice devs[8];
    vkEnumeratePhysicalDevices(inst, &n, devs);
    if (!n) die("no GPU");
    g_phys = devs[0];
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { g_phys = devs[i]; break; }
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_phys, &props);
    logf("GPU %s vendor=0x%x device=0x%x driver=0x%x api=0x%x", props.deviceName, props.vendorID, props.deviceID, props.driverVersion, props.apiVersion);
    fprintf(stderr, "GPU %s\n", props.deviceName);
    vkGetPhysicalDeviceMemoryProperties(g_phys, &g_memprops);

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(g_phys, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> exts(ne);
    vkEnumerateDeviceExtensionProperties(g_phys, nullptr, &ne, exts.data());
    auto has = [&](const char* e) { return std::any_of(exts.begin(), exts.end(), [&](auto& x) { return !strcmp(x.extensionName, e); }); };
    std::vector<const char*> enable;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat8FeaturesEXT ff8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT fi64{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT};
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV fcm2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV};
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR fcsd{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    void** tail = &f2.pNext;
    auto chain = [&](void* s) {   // (appends s to f2's chain; s's own pNext is where the next one goes)
        *tail = s;
        tail = (void**)&((VkBaseOutStructure*)s)->pNext;
    };
    chain(&f11), chain(&f12), chain(&f13);
    struct Opt { const char* name; void* feat; };
    for (Opt o : {Opt{"VK_KHR_cooperative_matrix", &fcm}, Opt{"VK_EXT_shader_float8", &ff8}, Opt{"VK_EXT_shader_image_atomic_int64", &fi64},
                  Opt{"VK_NV_cooperative_matrix2", &fcm2}, Opt{"VK_KHR_compute_shader_derivatives", &fcsd}, Opt{"VK_AMD_buffer_marker", nullptr},
                  Opt{"VK_AMD_device_coherent_memory", nullptr}, Opt{"VK_KHR_push_descriptor", nullptr}}) {
        const bool h = has(o.name);
        logf("extension %s: %s", o.name, h ? "yes" : "NO");
        if (!h) continue;
        enable.push_back(o.name);
        if (o.feat) chain(o.feat);
    }
    vkGetPhysicalDeviceFeatures2(g_phys, &f2);
    logf("features: coopmat=%d float8=%d float8coop=%d imageInt64=%d shaderInt64=%d float16=%d int8=%d varptr=%d", fcm.cooperativeMatrix, ff8.shaderFloat8,
         ff8.shaderFloat8CooperativeMatrix, fi64.shaderImageInt64Atomics, f2.features.shaderInt64, f12.shaderFloat16, f12.shaderInt8, f11.variablePointers);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &nq, qf.data());
    uint32_t family = 0;
    for (uint32_t i = 0; i < nq; i++)
        if ((qf[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { family = i; break; }
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family, qci.queueCount = 1, qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2, dci.queueCreateInfoCount = 1, dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)enable.size(), dci.ppEnabledExtensionNames = enable.data();
    if (VkResult r = vkCreateDevice(g_phys, &dci, nullptr, &g_device)) die("device %d", r);
#define LOADD(n) n = (PFN_##n)vkGetDeviceProcAddr(g_device, #n);
    DEVICE_FUNCS(LOADD)
    vkGetDeviceQueue(g_device, family, 0, &g_queue);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, pci.queueFamilyIndex = family;
    VkCommandPool pool;
    vkCreateCommandPool(g_device, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool, cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, cai.commandBufferCount = 1;
    vkAllocateCommandBuffers(g_device, &cai, &g_cmd);

    // ---- the provider
    void* lib = load_lib(argv[1]);
    if (!lib) die("%s did not load", argv[1]);
    auto create = (PfnFfxCreateContext)sym(lib, "ffxCreateContext");
    auto destroy = (PfnFfxDestroyContext)sym(lib, "ffxDestroyContext");
    auto configure = (PfnFfxConfigure)sym(lib, "ffxConfigure");
    auto query = (PfnFfxQuery)sym(lib, "ffxQuery");
    auto dispatch = (PfnFfxDispatch)sym(lib, "ffxDispatch");
    if (!create || !destroy || !configure || !query || !dispatch) die("the provider lacks an entry point");

    ffxConfigureDescGlobalDebug dbg{};
    dbg.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG;
    dbg.fpMessage = message, dbg.debugLevel = FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE;
    logf("configure(global debug) -> %u", configure(nullptr, &dbg.header));

    VkDeviceContext dc{g_device, g_phys, hook::GetDeviceProcAddr};
    uint64_t count = 0, ids[16]{};
    const char* names[16]{};
    ffxQueryDescGetVersions qv{};
    qv.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    qv.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION, qv.device = &dc, qv.outputCount = &count;
    ffxReturnCode_t r = query(nullptr, &qv.header);
    logf("query versions -> %u count=%llu", r, (unsigned long long)count);
    count = std::min<uint64_t>(count, 16);
    qv.versionIds = ids, qv.versionNames = names;
    r = query(nullptr, &qv.header);
    for (uint64_t i = 0; i < count; i++) {
        logf("  version %llu: id=0x%llx name=%s", (unsigned long long)i, (unsigned long long)ids[i], names[i] ? names[i] : "?");
        fprintf(stderr, "provider version: %s (0x%llx)\n", names[i] ? names[i] : "?", (unsigned long long)ids[i]);
    }

    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.deviceContext = dc;
    ffxOverrideVersion over{};
    over.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    over.header.pNext = &backend.header;
    if (const char* want = getenv("TRACE_VERSION")) {
        over.versionId = strtoull(want, nullptr, 0);
        logf("override version 0x%llx", (unsigned long long)over.versionId);
    }
    ffxCreateContextDescFrameGenerationVersion version{};
    version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
    version.header.pNext = over.versionId ? &over.header : &backend.header;
    version.version = FFX_FRAMEGENERATION_VERSION;
    ffxCreateContextDescFrameGeneration cc{};
    cc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    cc.header.pNext = &version.header;
    cc.flags = FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED | FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE;
    if (const char* fl = getenv("TRACE_CREATE_FLAGS")) cc.flags = strtoul(fl, nullptr, 0);
    cc.displaySize = {W, Hh}, cc.maxRenderSize = {RW, RH};
    const VkFormat color_format = getenv("TRACE_COLOR_FORMAT") ? (VkFormat)atoi(getenv("TRACE_COLOR_FORMAT")) : VK_FORMAT_R8G8B8A8_UNORM;
    cc.backBufferFormat = ffxApiGetSurfaceFormatVK(color_format);
    // (TRACE_HUD: a HUD in the presented colour, the HUD-less colour beside it; TRACE_HUDLESS_FORMAT: in its own format,
    // through the create description for it)
    const bool hud = getenv("TRACE_HUD") != nullptr;
    const VkFormat hudless_format = getenv("TRACE_HUDLESS_FORMAT") ? (VkFormat)atoi(getenv("TRACE_HUDLESS_FORMAT")) : color_format;
    ffxCreateContextDescFrameGenerationHudless hl{};
    if (getenv("TRACE_HUDLESS_FORMAT")) {
        hl.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS, hl.header.pNext = cc.header.pNext;
        hl.hudlessBackBufferFormat = ffxApiGetSurfaceFormatVK(hudless_format);
        cc.header.pNext = &hl.header;
    }
    ffxContext ctx = nullptr;
    logf("=== create context display=%ux%u render=%ux%u flags=0x%x format=%d", W, Hh, RW, RH, cc.flags, color_format);
    r = create(&ctx, &cc.header, nullptr);
    logf("=== create context -> %u ctx=%p", r, ctx);
    fprintf(stderr, "create context -> %u\n", r);
    if (r != FFX_API_RETURN_OK || !ctx) return 2;
    ffxQueryGetProviderVersion pv{};
    pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    if (query(&ctx, &pv.header) == FFX_API_RETURN_OK) {
        logf("provider in use: %s (0x%llx)", pv.versionName ? pv.versionName : "?", (unsigned long long)pv.versionId);
        fprintf(stderr, "in use: %s\n", pv.versionName ? pv.versionName : "?");
    }

    VkImage color = our_image(color_format, W, Hh, "in_color");
    VkImage depth = our_image(VK_FORMAT_R32_SFLOAT, RW, RH, "in_depth");
    const bool display_mv = (cc.flags & FFX_FRAMEGENERATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) != 0;
    const uint32_t MW = display_mv ? W : RW, MH = display_mv ? Hh : RH;
    VkImage motion = our_image(VK_FORMAT_R16G16_SFLOAT, MW, MH, "in_motion");
    VkImage made_all[4];
    for (int k = 0; k < 4; k++) {
        char nm[16];
        snprintf(nm, sizeof nm, k ? "out_made%d" : "out_made", k);
        made_all[k] = our_image(color_format, W, Hh, nm);
        begin();
        to_layout(made_all[k], VK_IMAGE_LAYOUT_GENERAL);
        submit();
    }
    auto res = [](VkImage im, VkFormat f, uint32_t w, uint32_t h, bool write) {
        FfxApiResourceDescription d{};
        d.type = FFX_API_RESOURCE_TYPE_TEXTURE2D, d.format = ffxApiGetSurfaceFormatVK(f);
        d.width = w, d.height = h, d.depth = 1, d.mipCount = 1, d.flags = FFX_API_RESOURCE_FLAGS_NONE;
        d.usage = FFX_API_RESOURCE_USAGE_READ_ONLY | (write ? FFX_API_RESOURCE_USAGE_UAV : 0);
        return ffxApiGetResourceVK((void*)im, d, write ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS : FFX_API_RESOURCE_STATE_GENERIC_READ);
    };

    const float cam_near = getenv("TRACE_NEAR") ? (float)atof(getenv("TRACE_NEAR")) : 0.1f;
    const float cam_far = getenv("TRACE_FAR") ? (float)atof(getenv("TRACE_FAR")) : 3.402823466e+38f;
    Scene scene{W, Hh, RW, RH, (cc.flags & FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED) != 0, (cc.flags & FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE) != 0,
                cam_near, cam_far, getenv("TRACE_FOV") ? (float)atof(getenv("TRACE_FOV")) : 1.0f, MW, MH};
    VkImage hudless = hud ? our_image(hudless_format, W, Hh, "in_hudless") : VK_NULL_HANDLE;
    // (TRACE_DISTORTION=k: a barrel distortion field at the display size, k times as strong; in UV, after - before)
    VkImage distortion = VK_NULL_HANDLE;
    if (const char* k = getenv("TRACE_DISTORTION")) {
        distortion = our_image(VK_FORMAT_R16G16_SFLOAT, W, Hh, "in_distortion");
        const float strength = (float)atof(k);
        std::vector<uint16_t> field((size_t)W * Hh * 2);
        for (uint32_t y = 0; y < Hh; y++)
            for (uint32_t x = 0; x < W; x++) {
                const float u = (x + 0.5f) / W - 0.5f, v = (y + 0.5f) / Hh - 0.5f, r2 = u * u + v * v;
                field[((size_t)y * W + x) * 2 + 0] = half(strength * u * r2);
                field[((size_t)y * W + x) * 2 + 1] = half(strength * v * r2);
            }
        upload(distortion, field.data(), field.size() * 2);
    }
    for (int frame = 0; frame < frames; frame++) {
        scene.make(frame);
        if (hud) {
            const std::vector<uint8_t> clean = in_format(scene.color, hudless_format);
            upload(hudless, clean.data(), clean.size());
            draw_hud(scene.color, W, Hh, frame);
        }
        const std::vector<uint8_t> texels = in_format(scene.color, color_format);
        upload(color, texels.data(), texels.size());
        upload(depth, scene.depth.data(), scene.depth.size() * 4);
        upload(motion, scene.motion.data(), scene.motion.size() * 2);
        logf("=== FRAME %d", frame);
        // (TRACE_IDS=a,b,c...: the frame IDs to use, frame by frame, to skip or repeat; otherwise 100, 101, ...)
        uint64_t id = 100 + frame;
        if (const char* ids = getenv("TRACE_IDS")) {
            const char* q = ids;
            for (int k = 0; k < frame && q; k++) q = strchr(q, ','), q = q ? q + 1 : nullptr;
            if (q) id = strtoull(q, nullptr, 10);
        }
        logf("FRAMEID %llu", (unsigned long long)id);
        ffxConfigureDescFrameGeneration cfg{};
        cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        cfg.frameGenerationEnabled = true;
        cfg.generationRect = {0, 0, (int32_t)W, (int32_t)Hh};
        if (const char* v = getenv("TRACE_RECT")) sscanf(v, "%d,%d,%d,%d", &cfg.generationRect.left, &cfg.generationRect.top, &cfg.generationRect.width, &cfg.generationRect.height);
        cfg.frameID = id;
        // (no swapchain of AMD's: without this flag Configure wants one and fails)
        cfg.flags = getenv("TRACE_CONFIG_FLAGS") ? strtoul(getenv("TRACE_CONFIG_FLAGS"), nullptr, 0) : FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;
        if (hud) cfg.HUDLessColor = res(hudless, hudless_format, W, Hh, false);
        r = configure(&ctx, &cfg.header);
        logf("=== configure -> %u", r);
        if (distortion) {
            ffxConfigureDescFrameGenerationRegisterDistortionFieldResource dc{};
            dc.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_REGISTERDISTORTIONRESOURCE;
            dc.distortionField = res(distortion, VK_FORMAT_R16G16_SFLOAT, W, Hh, false);
            r = configure(&ctx, &dc.header);
            logf("=== configure distortion -> %u", r);
        }
        begin();
        ffxDispatchDescFrameGenerationPrepareV2 prep{};
        prep.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
        prep.frameID = id, prep.commandList = (void*)g_cmd;
        prep.renderSize = {RW, RH};
        auto envf = [](const char* n, float d) { const char* v = getenv(n); return v ? (float)atof(v) : d; };
        auto env2 = [](const char* n, float& x, float& y) { if (const char* v = getenv(n)) sscanf(v, "%f,%f", &x, &y); };
        prep.jitterOffset = {0.0f, 0.0f};
        env2("TRACE_JITTER", prep.jitterOffset.x, prep.jitterOffset.y);
        if (frame & 1) prep.jitterOffset.x = -prep.jitterOffset.x * 0.5f, prep.jitterOffset.y *= 0.25f;
        prep.motionVectorScale = {1.0f, 1.0f};
        env2("TRACE_MVSCALE", prep.motionVectorScale.x, prep.motionVectorScale.y);
        prep.frameTimeDelta = envf("TRACE_FRAMETIME", 16.6667f);
        prep.reset = frame == 0 || (getenv("TRACE_RESET_AT") && frame == atoi(getenv("TRACE_RESET_AT")));
        prep.cameraNear = cam_near, prep.cameraFar = cam_far;
        prep.cameraFovAngleVertical = envf("TRACE_FOV", 1.0f);
        prep.viewSpaceToMetersFactor = envf("TRACE_V2M", 1.0f);
        prep.depth = res(depth, VK_FORMAT_R32_SFLOAT, RW, RH, false);
        prep.motionVectors = res(motion, VK_FORMAT_R16G16_SFLOAT, MW, MH, false);
        {   // the camera: still, or (TRACE_CAMERA=k) moving and turning; TRACE_CAMERA_API=still tells the provider it is still
            const Cam c = camera_at(getenv("TRACE_CAMERA_API") && !strcmp(getenv("TRACE_CAMERA_API"), "still") ? 0 : frame);
            const float *pos = c.pos, *rgt = c.right, *up = c.up, *f = c.fwd;
            for (int k = 0; k < 3; k++) prep.cameraForward[k] = f[k], prep.cameraRight[k] = rgt[k], prep.cameraUp[k] = up[k], prep.cameraPosition[k] = pos[k];
            logf("CAMERA pos=%.9g,%.9g,%.9g right=%.9g,%.9g,%.9g up=%.9g,%.9g,%.9g fwd=%.9g,%.9g,%.9g near=%.9g far=%.9g fov=%.9g jitter=%.9g,%.9g mvscale=%.9g,%.9g dt=%.9g v2m=%.9g",
                 pos[0], pos[1], pos[2], rgt[0], rgt[1], rgt[2], up[0], up[1], up[2], f[0], f[1], f[2], prep.cameraNear, prep.cameraFar,
                 prep.cameraFovAngleVertical, prep.jitterOffset.x, prep.jitterOffset.y, prep.motionVectorScale.x, prep.motionVectorScale.y,
                 prep.frameTimeDelta, prep.viewSpaceToMetersFactor);
        }
        logf("=== dispatch prepare");
        r = dispatch(&ctx, &prep.header);
        logf("=== dispatch prepare -> %u", r);
        ffxDispatchDescFrameGeneration fg{};
        fg.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
        fg.commandList = (void*)g_cmd;
        fg.presentColor = res(color, color_format, W, Hh, false);
        fg.numGeneratedFrames = getenv("TRACE_GEN") ? atoi(getenv("TRACE_GEN")) : 1;
        for (uint32_t k = 0; k < fg.numGeneratedFrames && k < 4; k++) fg.outputs[k] = res(made_all[k], color_format, W, Hh, true);
        fg.reset = frame == 0 || (getenv("TRACE_RESET_AT") && frame == atoi(getenv("TRACE_RESET_AT")));
        fg.backbufferTransferFunction = getenv("TRACE_TRANSFER") ? atoi(getenv("TRACE_TRANSFER")) : FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
        fg.minMaxLuminance[0] = 0.0f, fg.minMaxLuminance[1] = 1.0f;
        env2("TRACE_MINMAX", fg.minMaxLuminance[0], fg.minMaxLuminance[1]);
        fg.generationRect = cfg.generationRect;
        fg.frameID = id;
        logf("=== dispatch generate");
        r = dispatch(&ctx, &fg.header);
        logf("=== dispatch generate -> %u", r);
        fprintf(stderr, "frame %d: generate -> %u, %d dispatches so far\n", frame, r, g_dispatch);
        submit();
        logf("=== frame %d submitted and finished", frame);
        if (frame >= dump_from) {
            char prefix[32];
            snprintf(prefix, sizeof prefix, "f%d", frame);
            dump_images(prefix, !getenv("TRACE_DUMP_ALL") || !*getenv("TRACE_DUMP_ALL"));   // (unset or empty: only the harness's own images)
            if (getenv("TRACE_DUMP_BUFFERS")) dump_buffers(prefix, (VkDeviceSize)atoll(getenv("TRACE_DUMP_BUFFERS")));
        }
    }
    vkDeviceWaitIdle(g_device);
    r = destroy(&ctx, nullptr);
    logf("=== destroy -> %u", r);
    fprintf(stderr, "done: %d dispatches traced\n", g_dispatch);
    fclose(g_log);
    return 0;
}
