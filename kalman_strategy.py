"""
Kalman-Filtered Pairs Trading Strategy — MA / V
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

TICKER_Y = "MA"
TICKER_X = "V"

START_DATE = "2016-01-01"
END_DATE   = "2026-01-01"

DELTA  = 1e-4
VE     = 1e-3
WARMUP = 50

ENTRY_Z = 1
EXIT_Z  = 0
TC_BPS  = 2

PLOT = True


# ============================================================
# HELPERS
# ============================================================

def load_prices(ticker_y, ticker_x, start, end):
    """
    Load cached price data. Regenerates from yfinance if the cache is missing.
    """
    os.makedirs("data", exist_ok=True)

    y_path = f"data/{ticker_y}.csv"
    x_path = f"data/{ticker_x}.csv"

    if not (os.path.exists(y_path) and os.path.exists(x_path)):
        # Regenerate both from yfinance
        y = yf.download(ticker_y, start=start, end=end,
                        progress=False, multi_level_index=False)["Close"]
        x = yf.download(ticker_x, start=start, end=end,
                        progress=False, multi_level_index=False)["Close"]
        y.to_csv(y_path)
        x.to_csv(x_path)
    else:
        y = pd.read_csv(y_path, index_col=0, parse_dates=True)["Close"]
        x = pd.read_csv(x_path, index_col=0, parse_dates=True)["Close"]

    y, x = y.align(x, join="inner")
    return y, x

def print_stats(stats, header=None):
    if header:
        print(f"\n=== {header} ===")
    for k, v in stats.items():
        if k == "Sharpe":
            print(f"  {k:15s}: {v:>7.2f}")
        elif k == "N days":
            print(f"  {k:15s}: {v:>7d}")
        else:
            print(f"  {k:15s}: {v:>7.2%}")

# ============================================================
# KALMAN FILTER
# ============================================================

def kalman_filter(x, y, delta, Ve):
    """
    Dynamic linear regression via Kalman filter:
        y(t) = beta0(t) + beta1(t) * x(t) + eps(t)

    Returns:
        beta : (T, 2) — [intercept, slope]
        e    : (T,)   — forecast error (spread)
        Q    : (T,)   — forecast error variance
    """
    T = len(y)
    X = np.column_stack([np.ones(T), x])

    beta = np.zeros((T, 2))
    e    = np.zeros(T)
    Q    = np.zeros(T)
    P    = np.zeros((2, 2))
    Vw   = delta / (1 - delta) * np.eye(2)

    for t in range(T):
        if t > 0:
            beta[t] = beta[t - 1]
        R = P + Vw

        yhat = X[t] @ beta[t]
        Q[t] = X[t] @ R @ X[t] + Ve
        e[t] = y[t] - yhat

        K = R @ X[t] / Q[t]
        beta[t] = beta[t] + K * e[t]

        # Joseph form for numerical stability
        I = np.eye(2)
        KH = np.outer(K, X[t])
        P = (I - KH) @ R @ (I - KH).T + np.outer(K, K) * Ve

    return beta, e, Q


# ============================================================
# SIGNALS
# ============================================================

def bollinger_signals(zscore, entry_z, exit_z, warmup):
    """
    Position units in {-1, 0, +1} from a pre-computed z-score.

    NaN values (warm-up) and the first `warmup` bars are treated as
    'no signal' — the position stays flat.
    """
    T = len(zscore)
    pos = np.zeros(T)
    current = 0

    for t in range(warmup, T):
        z = zscore[t]
        if np.isnan(z):
            pos[t] = current
            continue

        if current == 0:
            if z < -entry_z:
                current = 1
            elif z > entry_z:
                current = -1
        elif current == 1:
            if z >= -exit_z:
                current = 0
        elif current == -1:
            if z <= exit_z:
                current = 0
        pos[t] = current

    return pos


# ============================================================
# BACKTEST
# ============================================================

def backtest(y, x, positions, beta, tc_bps):
    y_ret = y.pct_change().fillna(0).values
    x_ret = x.pct_change().fillna(0).values
    slope = beta[:, 1]

    # Lag positions by one bar to avoid look-ahead
    pos_lag = np.roll(positions, 1)
    pos_lag[0] = 0

    gross = pos_lag * (y_ret - slope * x_ret)

    trades = np.abs(np.diff(positions, prepend=0))
    cost = trades * 2 * tc_bps / 10000

    net = gross - cost

    return pd.DataFrame({
        "position":  positions,
        "gross_ret": gross,
        "cost":      cost,
        "net_ret":   net,
        "cum_net":   (1 + net).cumprod(),
    }, index=y.index)


# ============================================================
# STATS
# ============================================================

def performance_stats(returns, periods_per_year=252):
    r = returns.dropna()
    if len(r) == 0:
        return {}
    ann_ret = (1 + r).prod() ** (periods_per_year / len(r)) - 1
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + r).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return {
        "APR":          ann_ret,
        "Annual Vol":   ann_vol,
        "Sharpe":       sharpe,
        "Max Drawdown": dd.min(),
        "Total Return": cum.iloc[-1] - 1,
        "N days":       len(r),
    }


# ============================================================
# FIGURES
# ============================================================

def save_figures(y, beta, e, zscore, result, warmup):
    """Save the four-panel diagnostic chart plus the cumulative-return chart."""
    os.makedirs("figures", exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Z-score
    axes[0].plot(y.index, zscore, color="steelblue", lw=0.8)
    axes[0].axhline( ENTRY_Z, color="red", ls=":", alpha=0.6)
    axes[0].axhline(-ENTRY_Z, color="red", ls=":", alpha=0.6)
    axes[0].axhline(0, color="black", ls="--", alpha=0.5)
    axes[0].axvline(y.index[warmup], color="gray", ls="--", alpha=0.5)
    axes[0].set_ylabel("Z-score")

    # Panel 2: Hedge ratio
    axes[1].plot(y.index, beta[:, 1], color="steelblue")
    axes[1].axvline(y.index[warmup], color="gray", ls="--", alpha=0.5)
    axes[1].set_ylabel("Hedge ratio")

    # Panel 3: Spread
    axes[2].plot(y.index, e, color="steelblue", lw=0.8)
    axes[2].axhline(0, color="black", ls="--", alpha=0.5)
    axes[2].axvline(y.index[warmup], color="gray", ls="--", alpha=0.5)
    axes[2].set_ylabel("Spread")

    # Panel 4: Cumulative return
    axes[3].plot(result.index, result["cum_net"], color="steelblue")
    axes[3].axvline(y.index[warmup], color="gray", ls="--", alpha=0.5)
    axes[3].set_ylabel("Cumulative return")

    plt.tight_layout()
    fig.savefig("figures/kalman_panels.png", dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    y, x = load_prices(TICKER_Y, TICKER_X, START_DATE, END_DATE)
    print(f"Loaded {len(y)} bars from {y.index[0].date()} to {y.index[-1].date()}")

    beta, e, Q = kalman_filter(x.values, y.values, DELTA, VE)
    e[:WARMUP] = np.nan
    Q[:WARMUP] = np.nan
    zscore = e / np.sqrt(Q)
    positions = bollinger_signals(zscore, ENTRY_Z, EXIT_Z, WARMUP)

    # Headline result at the config TC_BPS
    result = backtest(y, x, positions, beta, TC_BPS)
    print_stats(performance_stats(result["net_ret"]), header=f"PERFORMANCE (TC = {TC_BPS} bps)")

    # Diagnostics
    n_trades = int(np.abs(np.diff(positions)).sum())
    days_in_pos = (positions != 0).sum()
    print(f"\nDays in position:   {days_in_pos} / {len(positions)} ({days_in_pos/len(positions):.1%})")
    print(f"Number of trades:   {n_trades}")
    print(f"Total cost:         {result['cost'].sum():.2%}")
    print(f"Z-score std:        {np.nanstd(zscore):.4f}")
    print(f"Z-score range:      [{np.nanmin(zscore):.2f}, {np.nanmax(zscore):.2f}]")

    # Cost sensitivity
    print("\n=== COST SENSITIVITY ===")
    print(f"  {'TC (bps)':<10} {'Net Return':>12} {'Sharpe':>10}")
    for tc in [0, 1, 2, 5, 10]:
        res_tc = backtest(y, x, positions, beta, tc)
        stats = performance_stats(res_tc["net_ret"])
        print(f"  {tc:<10} {stats['Total Return']:>11.2%} {stats['Sharpe']:>10.2f}")

    if PLOT:
        save_figures(y, beta, e, zscore, result, WARMUP)


if __name__ == "__main__":
    main()
