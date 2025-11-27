"""
ml_runner.py

runs the ml models to on the parsed data
"""

import sys
from pathlib import Path
from sign_data import get_sign_data
from ml_scripts.training import train_all_targets


def main():
    signData = get_sign_data()
    results = train_all_targets(signData, models_dir=Path("models"))

    print("\n=== ML Training Summary ===")
    for r in results:
        if r.get("status") == "ok":
            print(f"{r['target']:20s} | acc={r['acc']:.3f} | f1={r['f1_macro']:.3f} | {r['path']}")
        else:
            print(f"{r['target']:20s} | SKIP ({r.get('reason', 'no reason given')})")

if __name__ == "__main__":
    main()
