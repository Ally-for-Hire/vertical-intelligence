#!/bin/bash
# watcher: emits a line only on anomalies/state changes (stall, dead, traceback, low RAM, finished)
ROOT=$(cd "$(dirname "$0")/.." \&\& pwd); R=$ROOT/results; low=0; last=""; seen=""
emit(){ [ "$1" != "$last" ] && echo "$1" && last="$1"; }
while true; do
  st=$(python $ROOT/core/ctl.py status x1_36 2>/dev/null); p=$(echo "$st" | grep -c "RUNNING"); c=$(echo "$st" | head -1 | grep -o "CPU [0-9]*" | grep -o "[0-9]*"); f=$(echo "$st" | head -1 | grep -o "free [0-9]*" | grep -o "[0-9]*"); s="$p ${c:-50} ${f:-100}"
  [ -z "$s" ] && { sleep 60; continue; }
  set -- $s; p=${1:-0}; c=${2:-0}; f=${3:-100}
  if grep -qs Traceback $R/x1_36_run.log; then emit "ERROR traceback in a run log"; fi
  if [ -f $R/x1_36_run.json ]; then emit "DONE X1.36 finished"; exit 0; fi
  if [ "$p" -eq 0 ] && [ ! -f $R/x1_36_run.json ]; then dead=$((${dead:-0}+1)); [ $dead -ge 5 ] && emit "DEAD no experiment process, no x1_36 result"; else dead=0; fi
  if [ "$p" -gt 0 ] && [ "$c" -lt 15 ]; then low=$((low+1)); [ $low -ge 5 ] && emit "STALL cpu ${c}% for 5+ min with $p procs"; else low=0; fi
  [ "$f" -lt 12 ] && emit "RAM free ${f}%"
  sleep 60
done
