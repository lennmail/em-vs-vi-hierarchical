"""Run the simulation study.

uv run scripts/run_simulations.py --method em
uv run scripts/run_simulations.py --method vi              # slow (hours)
uv run scripts/run_simulations.py --method vi --quick      # smoke test
"""

import argparse
import json
from pathlib import Path

from emvi.simulation import run

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["em", "vi"], required=True)
parser.add_argument("--config", default=ROOT / "configs" / "simulation.json")
parser.add_argument("--out", default=ROOT / "results")
parser.add_argument("--scenarios", nargs="*", help="subset of scenario names")
parser.add_argument("--num-sims", type=int, help="override number of replicates")
parser.add_argument("--workers", type=int)
parser.add_argument("--quick", action="store_true", help="3 replicates, short VI runs")
args = parser.parse_args()

cfg = json.loads(Path(args.config).read_text())
if args.quick:
    cfg["num_sims"] = 3
    cfg["vi"].update(max_iter_known=2000, max_iter_mixture=2000, n_samples=200)
if args.num_sims:
    cfg["num_sims"] = args.num_sims

if __name__ == "__main__":
    run(args.method, cfg, Path(args.out), args.scenarios, args.workers)
