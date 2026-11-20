"""FinRL PPO agent on a 10-ETF universe.

Trains a PPO actor-critic on a custom Gym env where:
  - State  = (last 60 days of normalized closes per symbol) flattened.
  - Action = continuous portfolio weights in [0, 1] per symbol, softmaxed.
  - Reward = next-day log return - turnover cost (bps).

Runs end-to-end on a GPU machine. The trained policy is then evaluated
on the 2020-2026 holdout via the existing backtest_engine so the
metric definitions stay consistent with the backtest suite.

Setup (once):
    pip install "stable-baselines3[extra]>=2.3" "gymnasium>=0.29"

Run:
    python3 scripts/finrl_ppo_etf.py --train --eval

Outputs:
    finrl_outputs/ppo_etf.zip
    finrl_outputs/eval_report.json   (CAGR, Sharpe, MDD vs equal-weight)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYMBOLS = ["SPY", "QQQ", "TLT", "GLD", "IWM", "VTI", "EFA", "VWO", "DBC", "IEF"]


def _load_closes(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    import sys

    # Prefer pre-fetched parquet if present (so the run doesn't need
    # outbound network access to Alpaca).
    cache = ROOT / "finrl_outputs" / "closes_2014_2026.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        df.index = pd.to_datetime(df.index).date
        df = df[(df.index >= start) & (df.index <= end)]
        cols = [c for c in symbols if c in df.columns]
        return df[cols].dropna(how="all")

    sys.path.insert(0, str(ROOT))
    from price_data import fetch_daily_closes  # noqa: E402

    return fetch_daily_closes(symbols, start=start, end=end).dropna(how="all")


def _build_env_class():
    """Defer gym imports so this script can be import-checked without SB3."""
    import gymnasium as gym
    from gymnasium import spaces

    class EtfPortfolioEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, closes: pd.DataFrame, lookback: int = 60, cost_bps: float = 1.0):
            super().__init__()
            self.closes = (
                closes.apply(pd.to_numeric, errors="coerce")
                .astype(float)
                .ffill()
                .bfill()
                .dropna(axis=1, how="any")
            )
            if len(self.closes) <= int(lookback) + 2:
                raise ValueError(
                    f"closes has only {len(self.closes)} rows after cleaning, "
                    f"need > {int(lookback) + 2} for lookback={lookback}"
                )
            self.symbols = list(self.closes.columns)
            self.n = len(self.symbols)
            self.lookback = int(lookback)
            self.cost_bps = float(cost_bps)
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-10.0, high=10.0, shape=(self.lookback * self.n,), dtype=np.float32
            )
            self._t = self.lookback
            self._prev_w = np.zeros(self.n, dtype=np.float32)

        def _obs(self) -> np.ndarray:
            window = self.closes.iloc[self._t - self.lookback : self._t].values
            ret = np.log(window[1:] / np.clip(window[:-1], 1e-8, None))
            mean = ret.mean(axis=0, keepdims=True)
            std = ret.std(axis=0, keepdims=True) + 1e-8
            z = (ret - mean) / std
            # Pad first row with zeros to keep shape (lookback, n).
            obs = np.concatenate([np.zeros((1, self.n)), z], axis=0)
            return obs.astype(np.float32).flatten()

        def reset(self, *, seed=None, options=None):  # type: ignore[override]
            super().reset(seed=seed)
            self._t = self.lookback
            self._prev_w = np.zeros(self.n, dtype=np.float32)
            return self._obs(), {}

        def step(self, action):
            a = np.clip(action, 0.0, 1.0)
            s = a.sum()
            w = a / s if s > 1e-8 else np.zeros_like(a)
            # Reward = next-day log return on these weights minus turnover cost.
            p_today = self.closes.iloc[self._t].values
            p_next = self.closes.iloc[self._t + 1].values if self._t + 1 < len(self.closes) else p_today
            ret_vec = np.log(np.clip(p_next, 1e-8, None) / np.clip(p_today, 1e-8, None))
            port_ret = float((w * ret_vec).sum())
            turnover = float(np.abs(w - self._prev_w).sum())
            cost = self.cost_bps * 1e-4 * turnover
            reward = port_ret - cost
            self._prev_w = w.astype(np.float32)
            self._t += 1
            terminated = self._t >= len(self.closes) - 1
            return self._obs(), reward, terminated, False, {"weights": w.tolist()}

    return EtfPortfolioEnv


def _train(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    closes = _load_closes(args.symbols, args.train_start, args.train_end)
    EtfEnv = _build_env_class()
    env = DummyVecEnv([lambda: EtfEnv(closes, lookback=args.lookback, cost_bps=args.cost_bps)])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        device=args.device,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "ppo_etf.zip")
    print(f"saved policy to {out_dir / 'ppo_etf.zip'}")


def _evaluate(args: argparse.Namespace) -> None:
    import sys

    from stable_baselines3 import PPO

    sys.path.insert(0, str(ROOT))
    from backtest_engine import compute_metrics, run_open_to_open_backtest  # noqa: E402
    from price_data import fetch_daily_opens  # noqa: E402

    closes = _load_closes(args.symbols, args.eval_start, args.eval_end)
    opens = fetch_daily_opens(args.symbols, start=args.eval_start, end=args.eval_end).reindex_like(closes).ffill()

    EtfEnv = _build_env_class()
    env = EtfEnv(closes, lookback=args.lookback, cost_bps=args.cost_bps)
    model = PPO.load(Path(args.out_dir) / "ppo_etf.zip", device=args.device)

    obs, _ = env.reset()
    weights_history = []
    dates = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, _trunc, info = env.step(action)
        weights_history.append(info["weights"])
        dates.append(closes.index[env._t - 1])
        if term:
            break

    pos = pd.DataFrame(weights_history, index=pd.Index(dates, name="date"), columns=closes.columns)
    bt = run_open_to_open_backtest(opens=opens.loc[pos.index], positions=pos, cost_bps=args.cost_bps)
    m = compute_metrics(bt.daily["ret"])

    # Equal-weight benchmark.
    eq = pd.DataFrame(1.0 / len(closes.columns), index=pos.index, columns=closes.columns)
    eq_bt = run_open_to_open_backtest(opens=opens.loc[eq.index], positions=eq, cost_bps=args.cost_bps)
    em = compute_metrics(eq_bt.daily["ret"])

    out = {
        "ppo": m,
        "equal_weight": em,
        "eval_start": str(args.eval_start),
        "eval_end": str(args.eval_end),
        "symbols": list(closes.columns),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    ap.add_argument("--train-start", type=lambda s: dt.date.fromisoformat(s), default=dt.date(2018, 11, 1))
    ap.add_argument("--train-end", type=lambda s: dt.date.fromisoformat(s), default=dt.date(2022, 12, 31))
    ap.add_argument("--eval-start", type=lambda s: dt.date.fromisoformat(s), default=dt.date(2023, 1, 1))
    ap.add_argument("--eval-end", type=lambda s: dt.date.fromisoformat(s), default=dt.date(2026, 4, 30))
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=str(ROOT / "finrl_outputs"))
    args = ap.parse_args()

    if not (args.train or args.eval):
        ap.error("specify --train and/or --eval")
    if args.train:
        _train(args)
    if args.eval:
        _evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
