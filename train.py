import subprocess
from pathlib import Path

DATASET_DIR = Path("/workspace/AIMS/samples")
TRIALS = 50

samples = sorted(DATASET_DIR.glob("sample_*.ply"))

for ply in samples:
    prefix = ply.with_suffix("")

    print("===================================")
    print("Running:", prefix)

    cmd = [
        "python",
        "/workspace/AIMS/get_best.py",
        "--i", str(prefix),
        "--n_trials", str(TRIALS),
        "--study_name", prefix.name,
        "--storage", "sqlite:///poisson_opt.db"
    ]

    subprocess.run(cmd)