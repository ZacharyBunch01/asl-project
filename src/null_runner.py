"""
null_runner.py

Run null (baseline) models on the ASL dataset and print a summary.
"""

from utils.sign_data import get_sign_data
from ml_scripts.null import null_all_targets


def main():
    print("Loading data...")
    signData = get_sign_data()
    print(f"[INFO] Data shape: {signData.shape}")

    # Run null baselines for all default targets
    results = null_all_targets(signData)

    # Pretty summary
    print("\n=== Null Baseline Summary ===")
    for r in results:
        target = r["target"]
        if r.get("status") != "ok":
            reason = r.get("reason", "")
            print(f"{target:<18} | SKIPPED | {reason}")
        else:
            print(
                f"{target:<18} | "
                f"maj_acc={r['acc_majority']:.3f} | maj_f1={r['f1_majority']:.3f} | "
                f"strat_acc={r['acc_stratified']:.3f} | strat_f1={r['f1_stratified']:.3f}"
            )


if __name__ == "__main__":
    main()

