# fsr4fg-vk: a native Linux Vulkan implementation of FSR Frame Generation 4.0.2, over AMD's shaders and weights,
# which are extracted at build time from AMD's DLL (FSR SDK 2.4.0) and never committed.
#
#   cmake -B build -DFSR4FG_AMD_DLL=/path/to/amd_fidelityfx_framegeneration_vk.dll && cmake --build build
#
# Targets:
#   assets          build/gen/assets.bin, assets_gen.h and patched/ from AMD's DLL (tools/setup.py), when they are
#                   missing, when the scripts or data/model.json change, or when FSR4FG_AMD_DLL names another file
#   fsr4fg          build/amd_fidelityfx_framegeneration_vk.so
#   trace           build/trace, the native harness
#   trace_exe       build/trace.exe, the harness for AMD's DLL under Wine (with mingw-w64, if found), and a copy of
#                   AMD's DLL beside it for harness/run_amd.sh
cmake_minimum_required(VERSION 3.20)
project(fsr4fg-vk LANGUAGES CXX ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

set(FSR4FG_AMD_DLL "$ENV{FSR4FG_AMD_DLL}" CACHE FILEPATH
    "AMD's amd_fidelityfx_framegeneration_vk.dll from FSR SDK 2.4.0 (SHA-256 d002dabf...98b4)")
option(FSR4FG_BUILD_HARNESS "Build the native test harness (build/trace)" ON)
option(FSR4FG_BUILD_HARNESS_EXE "Build the Windows test harness (build/trace.exe) with mingw-w64, if found" ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_path(VULKAN_HEADERS_DIR vulkan/vulkan_core.h REQUIRED)
set(FFX_API_DIR ${CMAKE_SOURCE_DIR}/third_party/amd-fidelityfx-api)

# ---- assets, from AMD's DLL
set(GEN ${CMAKE_BINARY_DIR}/gen)
set(ASSETS ${GEN}/assets.bin ${GEN}/assets_gen.h)
if(FSR4FG_AMD_DLL)
    get_filename_component(dll "${FSR4FG_AMD_DLL}" ABSOLUTE)
    if(NOT EXISTS "${dll}")
        message(FATAL_ERROR "FSR4FG_AMD_DLL: ${dll} does not exist")
    endif()
    # (rewritten only when the path changes, so a new path reruns setup.py even if the file is older than the assets)
    file(CONFIGURE OUTPUT ${GEN}/dll_path.txt CONTENT "${dll}\n")
    file(GLOB setup_scripts CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tools/*.py)
    add_custom_command(OUTPUT ${ASSETS}
        COMMAND Python3::Interpreter -I ${CMAKE_SOURCE_DIR}/tools/setup.py --dll ${dll} --out ${GEN}
        DEPENDS ${dll} ${GEN}/dll_path.txt ${CMAKE_SOURCE_DIR}/data/model.json ${setup_scripts}
        COMMENT "Extracting AMD's shaders and weights from ${dll}"
        VERBATIM)
elseif(NOT EXISTS ${GEN}/assets.bin)
    message(FATAL_ERROR "Name AMD's DLL: -DFSR4FG_AMD_DLL=/path/to/amd_fidelityfx_framegeneration_vk.dll "
        "(FSR SDK 2.4.0, Kits/FidelityFX/signedbin/). The build reads AMD's shaders and weights from it.")
else()
    message(STATUS "FSR4FG_AMD_DLL not set: using the assets already in ${GEN}")
endif()
add_custom_target(assets DEPENDS ${ASSETS})

# ---- the library
add_library(fsr4fg SHARED src/fsr4fg.cpp src/assets.S)
add_dependencies(fsr4fg assets)
set_target_properties(fsr4fg PROPERTIES OUTPUT_NAME amd_fidelityfx_framegeneration_vk PREFIX ""
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set_source_files_properties(src/assets.S PROPERTIES OBJECT_DEPENDS ${GEN}/assets.bin)
set_source_files_properties(src/fsr4fg.cpp PROPERTIES OBJECT_DEPENDS ${GEN}/assets_gen.h)
target_include_directories(fsr4fg PRIVATE ${FFX_API_DIR} ${GEN})
target_compile_options(fsr4fg PRIVATE
    $<$<COMPILE_LANGUAGE:ASM>:-Wa,-I${GEN}>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-missing-field-initializers -Wno-deprecated-declarations -Wno-unknown-pragmas
        -ffp-contract=off>)   # (the camera matrices are computed on the CPU: no fused multiply-adds, whatever the compiler)
target_link_options(fsr4fg PRIVATE -Wl,--no-undefined -Wl,-z,defs)
target_link_libraries(fsr4fg PRIVATE ${CMAKE_DL_LIBS})

# ---- the harness: native, and for Windows to run AMD's DLL under Wine
set(HARNESS_FLAGS -O1 -g -ffp-contract=off -Wno-deprecated-declarations -Wno-attributes -Wno-unknown-pragmas)
if(FSR4FG_BUILD_HARNESS)
    add_executable(trace harness/trace.cpp)
    set_target_properties(trace PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    target_include_directories(trace PRIVATE ${FFX_API_DIR})
    target_compile_options(trace PRIVATE ${HARNESS_FLAGS})
    target_link_libraries(trace PRIVATE ${CMAKE_DL_LIBS})
endif()

find_program(MINGW_CXX x86_64-w64-mingw32-g++)
if(FSR4FG_BUILD_HARNESS_EXE AND MINGW_CXX)
    # (mingw gets its own copy of the Vulkan headers: the host's include directory would bring glibc's with it)
    set(mingw_inc ${CMAKE_BINARY_DIR}/mingw-include)
    file(COPY ${VULKAN_HEADERS_DIR}/vulkan ${VULKAN_HEADERS_DIR}/vk_video DESTINATION ${mingw_inc})
    file(GLOB_RECURSE ffx_headers CONFIGURE_DEPENDS ${FFX_API_DIR}/*.h)
    set(exe_outputs ${CMAKE_BINARY_DIR}/trace.exe)
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/trace.exe
        COMMAND ${MINGW_CXX} -std=c++20 ${HARNESS_FLAGS} -I${FFX_API_DIR} -I${mingw_inc} -static
                ${CMAKE_SOURCE_DIR}/harness/trace.cpp -o ${CMAKE_BINARY_DIR}/trace.exe
        DEPENDS ${CMAKE_SOURCE_DIR}/harness/trace.cpp ${ffx_headers}
        COMMENT "Building the Windows harness (trace.exe) with mingw-w64"
        VERBATIM)
    if(FSR4FG_AMD_DLL)
        list(APPEND exe_outputs ${CMAKE_BINARY_DIR}/amd_fidelityfx_framegeneration_vk.dll)
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/amd_fidelityfx_framegeneration_vk.dll
            COMMAND ${CMAKE_COMMAND} -E copy ${dll} ${CMAKE_BINARY_DIR}/amd_fidelityfx_framegeneration_vk.dll
            DEPENDS ${dll} ${GEN}/dll_path.txt VERBATIM)
    endif()
    add_custom_target(trace_exe ALL DEPENDS ${exe_outputs} ${ASSETS})
elseif(FSR4FG_BUILD_HARNESS_EXE)
    message(STATUS "x86_64-w64-mingw32-g++ not found: no trace.exe, so no runs of AMD's DLL")
endif()
