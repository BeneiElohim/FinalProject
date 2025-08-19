from pathlib import Path
from signal_engine.paths import FEAT_DIR, ROOT

def get_symbols(provided: list[str] | None) -> list[str]:
    """
    Smart symbol discovery. It's a utility, so it belongs in a utility file.
    """
    if provided:
        if provided == ["universe"]:
            cfile = ROOT / "candidates.txt"
            if not cfile.exists():
                return []
            return [ln.strip() for ln in cfile.read_text().splitlines() if ln.strip()]
        return provided

    # Fallback discovery
    cfile = ROOT / "candidates.txt"
    if cfile.exists():
        cands = [ln.strip() for ln in cfile.read_text().splitlines() if ln.strip()]
        if cands:
            return cands
            
    return sorted(p.stem for p in FEAT_DIR.glob("*.csv"))