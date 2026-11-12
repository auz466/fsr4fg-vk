#!/bin/sh
# run_amd.sh <run name> [W H RW RH frames dumpFrom]: AMD's DLL under Wine, traced into re/<run name>
#
# Needs build/trace.exe and the DLL beside it (cmake with mingw-w64 and FSR4FG_AMD_DLL), and the rewritten shaders that
# tools/setup.py puts in build/gen/patched. Settings, from the environment or from <repo>/local.env (not committed):
#   FSR4FG_WINEPREFIX   the Wine prefix to run in (default: $WINEPREFIX, else Wine's own default)
#   FSR4FG_BUILD        the build directory (default: <repo>/build)
#   FSR4FG_RE           where runs go (default: <repo>/re)
here=$(cd "$(dirname "$0")" && pwd); root=$(dirname "$here")
[ -f "$root/local.env" ] && . "$root/local.env"
build=${FSR4FG_BUILD:-$root/build}; out=${FSR4FG_RE:-$root/re}
[ -n "$FSR4FG_WINEPREFIX" ] && export WINEPREFIX="$FSR4FG_WINEPREFIX"
for f in trace.exe amd_fidelityfx_framegeneration_vk.dll gen/patched; do
    [ -e "$build/$f" ] || { echo "run_amd.sh: $build/$f is missing (build with mingw-w64 and -DFSR4FG_AMD_DLL)" >&2; exit 1; }
done
name=$1; shift
rm -rf "${out:?}/$name"; mkdir -p "$out/$name"
cd "$build" && WINEDEBUG=-all TRACE_PATCHES="Z:$build/gen/patched" TRACE_DUMP_ALL=${TRACE_DUMP_ALL-1} \
  timeout 900 wine trace.exe amd_fidelityfx_framegeneration_vk.dll "Z:$out/$name" "$@" 2>&1 | grep -v "radv is not"
