#!/usr/bin/env bash
# Resumable G1 step runner. State in $G1_DIR/state.json; exit 0 all gating steps done, 10 gate failed, 11 deferred for time, 3 locked.
# Usage: run_g1.sh [--from NAME | --only NAME | --checkpoint | --status]. G1_BUDGET_UNIT=s is a test hook (default m).
set -euo pipefail

G1_HOME=$(cd "$(dirname "$0")" && pwd)
G1_DIR=${G1_DIR:-/workspace/g1}
G1_STEPS=${G1_STEPS:-$G1_HOME/steps.tsv}
PY=${G1_PY:-/opt/venv/bin/python}
NOW_CMD=${G1_NOW_CMD:-date +%s}
UNIT=${G1_BUDGET_UNIT:-m}
export G1_HOME G1_DIR
mkdir -p "$G1_DIR/logs"

# pystate OP [ARGS]: all state.json reads and writes (atomic rewrite).
pystate() {
  "$PY" - "$G1_DIR" "$@" << 'PYEOF'
import json, os, sys, time
g1, op, *a = sys.argv[1:]
path = f"{g1}/state.json"
try:
    st = json.load(open(path))
except (OSError, ValueError):
    st = {}
st.setdefault("steps", {})
def save(obj, p):
    json.dump(obj, open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)
if op == "init":
    if "run_id" not in st:
        st.update(run_id=os.environ.get("G1_RUN_ID") or time.strftime("g1-%Y%m%dT%H%M%SZ", time.gmtime()),
                  pod_id=os.environ.get("RUNPOD_POD_ID", ""),
                  deadline_epoch=int(os.environ["POD_DEADLINE_EPOCH"]))
        save(st, path)
elif op == "status":
    print(st["steps"].get(a[0], {}).get("status", ""))
elif op == "reset":  # forget these steps (not done)
    for n in a:
        st["steps"].pop(n, None)
    save(st, path)
elif op == "set":  # NAME STATUS key=val...
    s = st["steps"].setdefault(a[0], {})
    s["status"] = a[1]
    for kv in a[2:]:
        k, v = kv.split("=", 1)
        s[k] = int(v) if k in ("start", "end", "exit_code") else v
    save(st, path)
elif op == "checkpoint":
    st["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(st, f"{g1}/checkpoint.json")
elif op == "summary":
    c = lambda x: sum(v.get("status") == x for v in st["steps"].values())
    run = next((k for k, v in st["steps"].items() if v.get("status") == "running"), "-")
    print(f"run_id={st.get('run_id', '-')} done={c('done')} failed={c('failed')} deferred={c('deferred_time')} running={run}")
PYEOF
}

# Read-only commands never take the lock: the watchdog calls --checkpoint while a step runs.
case "${1:-}" in
  --checkpoint) pystate checkpoint; sync; exit 0 ;;
  --status) pystate summary; exit 0 ;;
esac

exec 9> "$G1_DIR/.lock"
flock -n 9 || { echo "run_g1: another instance holds $G1_DIR/.lock" >&2; exit 3; }
: "${POD_DEADLINE_EPOCH:?POD_DEADLINE_EPOCH is required}"
pystate init

mode=all target=
case "${1:-}" in
  --from) mode=from target=${2:?--from needs a step name} ;;
  --only) mode=only target=${2:?--only needs a step name} ;;
  "") ;;
  *) echo "usage: run_g1.sh [--from NAME | --only NAME | --checkpoint | --status]" >&2; exit 2 ;;
esac

started=
while IFS=$'\t' read -r name budget gating cmd; do
  case "$name" in "" | "#"*) continue ;; esac
  case $mode in
    from) [ -n "$started" ] || [ "$name" = "$target" ] || continue; started=1; pystate reset "$name" ;;
    only) [ "$name" = "$target" ] || continue; pystate reset "$name" ;;
  esac
  [ "$(pystate status "$name")" != "done" ] || continue

  now=$(eval "$NOW_CMD")
  if [ $((POD_DEADLINE_EPOCH - now)) -lt $(((budget + 20) * 60)) ]; then
    echo "run_g1: $name deferred: budget ${budget}m + 20m exceeds the time left" >&2
    pystate set "$name" deferred_time
    exit 11
  fi

  log=$G1_DIR/logs/$name.log
  pystate set "$name" running "start=$now" "log=$log"
  set +e
  timeout "${budget}$UNIT" bash -lc "$cmd" 9>&- 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -eq 0 ]; then st="done"; else st=failed; fi
  pystate set "$name" "$st" "end=$(eval "$NOW_CMD")" "exit_code=$rc"
  if [ "$st" = failed ] && [ "$gating" = 1 ]; then
    echo "run_g1: gating step $name failed (exit $rc)" >&2
    exit 10
  fi
done < "$G1_STEPS"
exit 0
