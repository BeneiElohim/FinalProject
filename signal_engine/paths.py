from pathlib import Path

ROOT        = Path(__file__).resolve().parents[1]

DATA        = ROOT / "data"
RAW_DIR     = DATA / "raw"
PROC_DIR    = DATA / "processed"
FEAT_DIR    = DATA / "features"
STRAT_DIR   = DATA / "strategies"
LOG_DIR     = ROOT / "logs"

for p in (RAW_DIR, PROC_DIR, FEAT_DIR, STRAT_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)
