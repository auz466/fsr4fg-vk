#!/bin/sh
# validation_all.sh: tests/validation.sh over the default inputs and two sets of features (motion vector flags,
# HUD-less colour in its own format, debug flags, BGRA back buffer). Fails if any run does.
here=$(dirname "$0"); rc=0
"$here/validation.sh" || rc=1
TRACE_CREATE_FLAGS=28 TRACE_JITTER=0.3,-0.2 TRACE_HUD=1 TRACE_CONFIG_FLAGS=12 "$here/validation.sh" || rc=1
TRACE_CREATE_FLAGS=26 TRACE_CONFIG_FLAGS=11 TRACE_HUD=1 TRACE_HUDLESS_FORMAT=37 TRACE_COLOR_FORMAT=44 "$here/validation.sh" || rc=1
exit $rc
