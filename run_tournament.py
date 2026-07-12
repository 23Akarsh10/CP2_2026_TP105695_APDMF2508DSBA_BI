import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from engine_final import fetch_data, add_features, run_model_tournament, select_champion, CONFIG, BURSA_10

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

MODEL_TYPES = ("XGBoost", "LSTM", "GRU")
N_SPLITS = CONFIG.n_splits          # walk-forward folds, default 5
USE_OPTUNA = True                   # falls back automatically if optuna isn't installed
XGB_TRIALS = 15
KERAS_TRIALS = 10

REPORT_DIR = Path("tournament_reports")


def main():
    per_stock = {}   # name -> tournament dict (results, tuned_params, champion, champion_params)
    failed = []

    for i, (name, symbol) in enumerate(BURSA_10.items(), start=1):
        print(f"[{i}/{len(BURSA_10)}] {name} ({symbol})")
        try:
            df = add_features(fetch_data(symbol))
            print(f"  {len(df)} rows ready. Running tournament "
                  f"({', '.join(MODEL_TYPES)}, {N_SPLITS}-fold walk-forward)...")
            tournament = run_model_tournament(
                df,
                model_types=MODEL_TYPES,
                n_splits=N_SPLITS,
                use_optuna=USE_OPTUNA,
                xgb_trials=XGB_TRIALS,
                keras_trials=KERAS_TRIALS,
            )
            per_stock[name] = tournament
            for mtype in MODEL_TYPES:
                r = tournament["results"][mtype]
                print(f"    {mtype:<10} MAE={r['avg_mae']:.4f}  Dir.Acc={r['avg_acc'] * 100:.1f}%  "
                      f"Sharpe={r['avg_sharpe']:.3f}")
        except Exception as exc:
            print(f"  SKIPPED ({name}): {exc}")
            failed.append(name)

    if not per_stock:
        print("\nNo stocks completed successfully - nothing to aggregate.")
        return 1

    # ── Aggregate: average MAE per model type across all stocks ─────
    avg_mae_by_model = {}
    avg_acc_by_model = {}
    avg_sharpe_by_model = {}
    for mtype in MODEL_TYPES:
        maes = [t["results"][mtype]["avg_mae"] for t in per_stock.values() if mtype in t["results"]]
        accs = [t["results"][mtype]["avg_acc"] for t in per_stock.values() if mtype in t["results"]]
        sharpes = [t["results"][mtype]["avg_sharpe"] for t in per_stock.values() if mtype in t["results"]]
        avg_mae_by_model[mtype] = float(np.mean(maes)) if maes else float("nan")
        avg_acc_by_model[mtype] = float(np.mean(accs)) if accs else float("nan")
        avg_sharpe_by_model[mtype] = float(np.mean(sharpes)) if sharpes else float("nan")

    # Champion is picked on a weighted composite of MAE (error, lower is
    # better), Directional Accuracy, and Sharpe Ratio (risk-adjusted
    # return, higher is better), all averaged across all stocks - not on
    # MAE alone.
    champion = select_champion({
        mtype: {
            "avg_mae": avg_mae_by_model[mtype],
            "avg_acc": avg_acc_by_model[mtype],
            "avg_sharpe": avg_sharpe_by_model[mtype],
        }
        for mtype in MODEL_TYPES
    })

    # Best single tuned instance of the champion architecture (lowest
    # MAE among the 10 stocks) - used as the actual hyperparameters,
    # since averaging hyperparameters across stocks isn't meaningful.
    best_stock_name, best_stock_mae = None, float("inf")
    for name, t in per_stock.items():
        mae = t["results"][champion]["avg_mae"]
        if mae < best_stock_mae:
            best_stock_mae = mae
            best_stock_name = name
    champion_params = per_stock[best_stock_name]["tuned_params"][champion]

    # ── Print aggregate report ───────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"AGGREGATE REPORT — average across {len(per_stock)}/{len(BURSA_10)} stocks")
    print("=" * 70)
    print(f"{'Model':<10}{'Avg MAE (RM)':>16}{'Avg Dir. Accuracy':>20}{'Avg Sharpe':>14}")
    print("-" * 70)
    for mtype in MODEL_TYPES:
        marker = "  <- champion" if mtype == champion else ""
        print(f"{mtype:<10}{avg_mae_by_model[mtype]:>16.4f}{avg_acc_by_model[mtype] * 100:>19.1f}%"
              f"{avg_sharpe_by_model[mtype]:>14.3f}{marker}")
    print("=" * 70)
    if failed:
        print(f"Skipped (data/fetch errors): {', '.join(failed)}")
    print(f"\nChampion architecture: {champion}")
    print(f"(selected on a weighted composite of avg MAE, avg Directional "
          f"Accuracy, and avg Sharpe Ratio across all stocks, not MAE alone)")
    print(f"Champion hyperparameters taken from: {best_stock_name} "
          f"(its best MAE for {champion} = {best_stock_mae:.4f})")
    print(json.dumps(champion_params, indent=2))

    # ── Save full report to disk ─────────────────────────────────────
    REPORT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"all_stocks_{timestamp}.json"

    per_stock_serialisable = {
        name: {
            "results": {
                mtype: {
                    "avg_mae": t["results"][mtype]["avg_mae"],
                    "avg_acc": t["results"][mtype]["avg_acc"],
                    "avg_sharpe": t["results"][mtype]["avg_sharpe"],
                }
                for mtype in MODEL_TYPES if mtype in t["results"]
            },
            "tuned_params": t["tuned_params"],
        }
        for name, t in per_stock.items()
    }
    report_path.write_text(json.dumps({
        "timestamp": timestamp,
        "n_splits": N_SPLITS,
        "stocks_run": list(per_stock.keys()),
        "stocks_skipped": failed,
        "per_stock": per_stock_serialisable,
        "avg_mae_by_model": avg_mae_by_model,
        "avg_acc_by_model": avg_acc_by_model,
        "avg_sharpe_by_model": avg_sharpe_by_model,
        "champion": champion,
        "champion_params_source_stock": best_stock_name,
        "champion_params": champion_params,
    }, indent=2))
    print(f"\nFull report saved to: {report_path}")

    # ── Ready-to-paste block for app.py ──────────────────────────────
    print("\n" + "=" * 70)
    print("Paste this at the top of app.py (replacing the existing")
    print("CHAMPION_MODEL / CHAMPION_PARAMS constants):")
    print("=" * 70)
    print(f'CHAMPION_MODEL = "{champion}"')
    print("CHAMPION_PARAMS = " + json.dumps(champion_params, indent=4))
    return 0


if __name__ == "__main__":
    sys.exit(main())