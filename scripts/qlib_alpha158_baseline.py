"""F-1: Qlib LightGBM Alpha158 baseline.

This script trains Qlib's reference Alpha158 + LightGBM model on the
SP500 / NASDAQ100 universe and reports IC / Rank-IC / cumulative
return on the 2020-2026 holdout.

It does NOT replace the built-in strategies; it produces an out-of-band
baseline so we can compare hand-engineered alpha against a
well-known industry benchmark.

Prerequisite (run once):
    1. The Qlib US daily data archive must be on disk at
       ~/.qlib/qlib_data/us_data. Fetch it from a machine with GitHub
       access:

           python3 -c "from qlib.tests.data import GetData; \\
               GetData().qlib_data(target_dir='~/.qlib/qlib_data/us_data', \\
                                   region='us', interval='1d', delete_old=True)"

    2. pip install lightgbm  (already pulled in by pyqlib)

Run:
    python3 scripts/qlib_alpha158_baseline.py

CPU only. Takes ~10-15 min for the default config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--qlib-data",
        default="~/.qlib/qlib_data/us_data",
        help="Path to Qlib's binary US data store",
    )
    ap.add_argument("--market", default="csi300", help="Universe name; for US data try 'sp500' or 'nasdaq100'")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default="2026-04-30")
    ap.add_argument("--train-end", default="2017-12-31")
    ap.add_argument("--valid-end", default="2019-12-31")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "qlib_outputs"))
    args = ap.parse_args()

    qdir = Path(args.qlib_data).expanduser()
    if not qdir.exists():
        print(f"ERROR: Qlib data directory not found at {qdir}")
        print("Follow the download instructions in this script's docstring.")
        return 2

    import qlib
    from qlib.config import REG_US
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord

    qlib.init(provider_uri=str(qdir), region=REG_US)

    handler_cfg = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": args.start,
            "end_time": args.end,
            "fit_start_time": args.start,
            "fit_end_time": args.train_end,
            "instruments": args.market,
        },
    }
    dataset_cfg = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_cfg,
            "segments": {
                "train": (args.start, args.train_end),
                "valid": (
                    str((Path(args.train_end))).split("T")[0] if False else args.train_end,
                    args.valid_end,
                ),
                "test": (args.valid_end, args.end),
            },
        },
    }
    model_cfg = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 8,
        },
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = init_instance_by_config(dataset_cfg)
    model = init_instance_by_config(model_cfg)

    with R.start(experiment_name="alpha158_lgb"):
        model.fit(dataset)
        R.save_objects(**{"model.pkl": model, "dataset.pkl": dataset})
        recorder = R.get_recorder()
        # Signal record: predictions on the test segment.
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        # Signal analysis: IC / Rank-IC summaries.
        sar = SigAnaRecord(recorder)
        sar.generate()
        # Portfolio analysis (top-30 long-only on the test segment).
        par = PortAnaRecord(recorder, config={"strategy": {"class": "TopkDropoutStrategy",
                                                            "module_path": "qlib.contrib.strategy.signal_strategy",
                                                            "kwargs": {"signal": "<PRED>", "topk": 30, "n_drop": 5}},
                                                "backtest": {"start_time": args.valid_end,
                                                             "end_time": args.end,
                                                             "account": 100000000,
                                                             "benchmark": "SH000300" if False else "^GSPC",
                                                             "exchange_kwargs": {"freq": "day", "limit_threshold": 0.095,
                                                                                 "deal_price": "close",
                                                                                 "open_cost": 0.0005,
                                                                                 "close_cost": 0.0015,
                                                                                 "min_cost": 5}}})
        par.generate()

    print(f"OK - artifacts under {out} and mlflow experiment 'alpha158_lgb'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
