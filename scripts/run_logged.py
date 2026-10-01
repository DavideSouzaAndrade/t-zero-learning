"""Run train.py with wandb.log redirected to a local JSONL file.

Usage: python scripts/run_logged.py OUT.jsonl --config a2c_cartpole --override ...
Used because wandb.ai is not reachable from the sandbox where the
experiments were run; the metric names are exactly the wandb ones.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["WANDB_MODE"] = "disabled"

import wandb  # noqa: E402

out_path = sys.argv.pop(1)
_fh = open(out_path, "w")


def _log(d, step=None, **_):
    row = {k: (float(v) if hasattr(v, "__float__") else v) for k, v in d.items()}
    if step is not None:
        row["_step"] = int(step)
    _fh.write(json.dumps(row) + "\n")
    _fh.flush()


wandb.log = _log
_orig_init = wandb.init


def _init(*a, **k):
    k["mode"] = "disabled"
    run = _orig_init(*a, **k)
    wandb.log = _log  # wandb.init rebinds the module-level log; patch again
    return run


wandb.init = _init

import train  # noqa: E402

train.main()
