#!/usr/bin/env bash
# monitor.sh <round> — poll the box, rsync the round's logs/results down, emit NEW
# progress and failure lines only. Network failures are swallowed so a blip does not
# end the watch; they are reported once per blip instead of silently.
round=${1:?round, e.g. r1}
box=adhavan_23bce0839@10.30.161.19
remote=research/framework/experiment/$round
local_dir=$(dirname "$0")/$round
# persist across re-arms: a fresh mktemp replays the whole log every time the
# 30-minute Monitor expires and is restarted
seen="$local_dir/.seen_box1"
touch "$seen"
down=0
# A round folder often holds ROUND DONE from an EARLIER pass (a rerun, a second
# script). Exiting on any occurrence kills the watch instantly; only a NEW one counts.
done0=$(grep -c "ROUND DONE" "$local_dir/logs/_progress.log" 2>/dev/null || echo 0)
while true; do
  if timeout 60 rsync -az "$box:$remote/logs/" "$local_dir/logs/" 2>/dev/null; then
    timeout 60 rsync -az "$box:$remote/results/" "$local_dir/results/" 2>/dev/null
    [ "$down" = 1 ] && echo "[net] box reachable again" && down=0
    # round starts/exits and anything that looks like a failure. Eval lines are
    # deliberately NOT matched: they arrive several times a minute across two
    # boxes and drown the channel. Recall progress comes from the 30-minute cron
    # check instead, which is what it is for.
    # -H keeps the log's name on each line: step lines from four runs are otherwise
    # indistinguishable, and dedup would be ambiguous across them.
    # box-2 logs (*_box2.log) live in the same folder and have their own monitor; skip them here
    grep -HE "^=== |Traceback|Error|error:|CUDA|out of memory|Killed|nan" \
        $(ls "$local_dir"/logs/*.log 2>/dev/null | grep -v '_box2\.log$') 2>/dev/null |
      sed "s#^$local_dir/logs/##" |
      while IFS= read -r line; do
        grep -qxF "$line" "$seen" || { echo "$line"; echo "$line" >> "$seen"; }
      done
    now=$(grep -c "ROUND DONE" "$local_dir/logs/_progress.log" 2>/dev/null || echo 0)
    [ "$now" -gt "$done0" ] && {
      echo "[$round] ROUND DONE — logs and results synced to $local_dir"; exit 0; }
  else
    [ "$down" = 0 ] && echo "[net] box unreachable, retrying every 60s" && down=1
  fi
  sleep 60
done
