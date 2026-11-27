#!/usr/bin/env python3
"""Write $G1_DIR/g1_summary.md from state.json, hostinfo.json, vram.csv, pins.txt and the junit XMLs."""

import csv
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

g1 = Path(os.environ.get("G1_DIR", "/workspace/g1"))
RATE = 0.352  # USD per pod hour, GPU plus disk


def load(name, default):
    try:
        return json.loads((g1 / name).read_text())
    except (OSError, ValueError):
        return default


def junit(name):
    try:
        root = ET.parse(g1 / name).getroot()
    except (OSError, ET.ParseError):
        return None
    s = root if root.tag == "testsuite" else root.find("testsuite")
    return {k: int(s.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}


state = load("state.json", {})
steps = state.get("steps", {})
status = lambda prefix: next(
    (v.get("status", "-") for k, v in steps.items() if k.startswith(prefix)), "not run"
)
host = load("hostinfo.json", {})

gate = [k for k in steps if k[:2] in ("02", "03", "04", "05")]
not_done = [p for p in ("02_", "03_", "04_", "05_") if status(p) != "done"]
names = {p: next((k for k in steps if k.startswith(p)), p) for p in not_done}
go = not not_done

now = int(os.environ.get("G1_NOW") or time.time())
hours = (now - (state.get("deadline_epoch", now) - 36000)) / 3600

pins = (g1 / "pins.txt").read_text().strip() if (g1 / "pins.txt").exists() else "(missing)"
nc, c = junit("pytest_not_carla.xml"), junit("pytest_carla.xml")
fmt = lambda j: "not run" if j is None else ", ".join(f"{k} {v}" for k, v in j.items())
try:
    vram = list(csv.reader((g1 / "vram.csv").open()))
except OSError:
    vram = []

L = ["# G1 summary", ""]
L += [f"Verdict: {'**GO**' if go else '**NO-GO**'}", ""]
if not go:
    L += [f"Not done: {', '.join(names[p] + ' (' + status(p) + ')' for p in not_done)}", ""]
L += [
    f"Run `{state.get('run_id', '-')}`, pod {state.get('pod_id', '-')}: {hours:.2f} h, ${hours * RATE:.2f} so far.",
    "",
]
L += ["## Exit check", "", "| Check | Result |", "|---|---|"]
L += [f"| Vulkan GO (steps 02-05) | {'GO' if go else 'NO-GO'} |"]
L += [f"| vram.csv measured | {len(vram) - 1 if vram else 0} rows |"]
L += [f"| Pins verified (step 07) | {status('07_')} |"]
L += [f"| pytest -m 'not carla' | {fmt(nc)} |"]
L += [f"| pytest -m carla / smoke 05-06 | {fmt(c)}; smoke 05-06 {status('11_')} |"]
L += ["| Auto-stop | on-pod watchdog and Mac watcher (flags: see status.json) |", ""]
L += ["## Steps", "", "| Step | Status | Exit |", "|---|---|---|"]
L += [f"| {k} | {v.get('status')} | {v.get('exit_code', '')} |" for k, v in steps.items()]
L += ["", "## Host", "", "```", json.dumps(host, indent=1), "```", ""]
if vram:
    L += ["## VRAM / tick rate", ""] + [f"| {' | '.join(r)} |" for r in vram[:1]]
    L += ["|" + "---|" * len(vram[0])] + [f"| {' | '.join(r)} |" for r in vram[1:]] + [""]
L += ["## Pins", "", "```", pins, "```", ""]
ecc = [k for k, v in steps.items() if v.get("status") == "failed" and k not in gate]
L += ["## ECC items (failed non-gating steps)", ""] + ([f"- {k}" for k in ecc] or ["- none"]) + [""]
(g1 / "g1_summary.md").write_text("\n".join(L))
