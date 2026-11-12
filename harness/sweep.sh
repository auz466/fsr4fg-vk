#!/bin/sh
# traces at several sizes, two frames each, no dumps: for fitting how sizes and constants follow the resolution
for s in "1280 720 1280 720" "2560 1440 2560 1440" "3840 2160 3840 2160" "2048 1152 2048 1152" "3440 1440 3440 1440" "2752 1152 2752 1152" "1600 900 1600 900" "1920 1080 1280 720" "1920 1080 960 540" "2048 1152 1364 768" "1024 576 1024 576" "5120 2880 5120 2880" "1920 1200 1920 1200" "1001 777 1001 777"; do
  set -- $s
  TRACE_DUMP_ALL= ./run_amd.sh "sz_$1x$2_$3x$4" $1 $2 $3 $4 2 99 | tail -1
done
