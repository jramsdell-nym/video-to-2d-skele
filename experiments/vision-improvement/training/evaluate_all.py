"""Serial local inference and scoring for the fixed, completed bat pair."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", type=Path, default=ROOT / "player-holdout-clean-20261111")
a = p.parse_args()
run = a.run.resolve()
for recipe in ("baseline", "candidate"):
    assert json.loads((run / f"{recipe}-status.json").read_text())["state"] == "complete"
provenance = json.loads((run / "execution-provenance.json").read_text())
for filename, snapshot in (("run_experiment.py", "runner-source-v3.py"), ("analyze_experiment.py", "analyzer-source-v1.py")):
    assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == provenance[snapshot]["sha256"]
assert hashlib.sha256((ROOT.parent / "quality" / "score_bat.py").read_bytes()).hexdigest() == provenance["scorer-source-v1.py"]["sha256"]
commands = []
for recipe, split in (("baseline", "external"), ("candidate", "external"), ("deployed", "external"), ("baseline", "val"), ("candidate", "val"), ("baseline", "test"), ("candidate", "test")):
    commands.append([sys.executable, str(ROOT / "run_experiment.py"), "predict", "--run", str(run), "--recipe", recipe, "--split", split])
commands.append([sys.executable, str(ROOT / "analyze_experiment.py"), "--run", str(run)])
receipt = {"startedAt": datetime.now(timezone.utc).isoformat(), "evaluatorSHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "commands": [], "state": "running"}
for command in commands:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, check=True)
    receipt["commands"].append({"argv": command, "completedAt": datetime.now(timezone.utc).isoformat()})
    (run / "evaluation-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
receipt["state"] = "complete"
receipt["completedAt"] = datetime.now(timezone.utc).isoformat()
(run / "evaluation-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
