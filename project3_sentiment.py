"""
================================================================================
PROJECT 3 — SENTIMENT + PRICE MODEL (Alternative Data)
================================================================================
  1. Simulated news headline sentiment (VADER-style lexicon scoring)
  2. Sentiment momentum vs price momentum signal fusion
  3. Sentiment regime classification
  4. Lead-lag analysis: does sentiment lead price?
  5. Full backtest: sentiment-only, price-only, combined
  6. Sentiment decay model (relevance half-life)
  7. Event study: sentiment spike analysis
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import correlate
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

DARK   = "#0f172a"; CARD   = "#1e293b"; ACCENT = "#3b82f6"
GREEN  = "#10b981"; AMBER  = "#f59e0b"; RED    = "#ef4444"
LIGHT  = "#e2e8f0"; MUTED  = "#64748b"; GRID   = "#334155"
PURPLE = "#8b5cf6"; TEAL   = "#14b8a6"

COMMISSION = 0.001
SLIPPAGE   = 0.0005
INITIAL    = 100_000


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SIMULATE NEWS SENTIMENT DATA
# ═══════════════════════════════════════════════════════════════════════════════

HEADLINE_TEMPLATES = {
    "strong_positive": [
        "Company beats earnings by wide margin, raises guidance",
        "Record revenue reported, shares surge in pre-market",
        "Major acquisition approved, analysts upgrade to buy",
        "Breakthrough product launch drives massive demand",
        "CEO announces share buyback program worth billions",
    ],
    "mild_positive": [
        "Company meets expectations, maintains outlook",
        "Quarterly results in line with analyst estimates",
        "New partnership announced with industry leader",
        "Cost-cutting measures show early positive results",
    ],
    "neutral": [
        "Company schedules earnings call for next quarter",
        "Management changes announced at board level",
        "Regulatory filing submitted as expected",
        "Annual shareholder meeting confirms existing strategy",
    ],
    "mild_negative": [
        "Company misses estimates slightly, lowers guidance",
        "Supply chain challenges expected in coming quarter",
        "Competition intensifies in core market segment",
        "Analyst downgrades to neutral citing valuation",
    ],
    "strong_negative": [
        "Major earnings miss triggers selloff, CEO under pressure",
        "Regulatory investigation launched, shares plunge",
        "Product recall announced, liability concerns mount",
        "Credit downgrade issued, debt costs rising sharply",
        "Fraud allegations surface, trading halted briefly",
    ],
}

SENTIMENT_SCORES = {
    "strong_positive":  0.85,
    "mild_positive":    0.40,
    "neutral":          0.00,
    "mild_negative":   -0.40,
    "strong_negative": -0.85,
}


def simulate_sentiment_and_price(n_days=1500, sentiment_lead=2,
                                  noise_level=0.6, seed=42):
    """
    Simulate:
    - Underlying 'true' sentiment signal
    - Noisy observed sentiment (from scraping headlines)
    - Price that partially reflects sentiment with a lag
    - Random news events
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)

    # True latent sentiment (slow-moving AR process)
    true_sent  = np.zeros(n_days)
    true_sent[0] = 0
    for i in range(1, n_days):
        true_sent[i] = 0.92 * true_sent[i-1] + 0.08 * rng.standard_normal()

    # Occasional sentiment shocks (earnings, news events)
    n_events = n_days // 60
    event_idx = rng.choice(n_days, n_events, replace=False)
    for idx in event_idx:
        shock = rng.choice([-1.5, -1.0, 1.0, 1.5],
                           p=[0.2, 0.3, 0.3, 0.2])
        window = min(10, n_days - idx)
        true_sent[idx:idx+window] += shock * np.exp(-np.arange(window) * 0.3)

    true_sent = np.clip(true_sent, -3, 3)

    # Observed sentiment = true + noise (scraping is noisy)
    obs_sent = true_sent + noise_level * rng.standard_normal(n_days)
    obs_sent = np.clip(obs_sent, -3, 3)

    # Assign headline categories
    def score_to_category(s):
        if s > 1.2:  return "strong_positive"
        if s > 0.3:  return "mild_positive"
        if s > -0.3: return "neutral"
        if s > -1.2: return "mild_negative"
        return "strong_negative"

    categories = [score_to_category(s) for s in obs_sent]
    headlines  = [rng.choice(HEADLINE_TEMPLATES[c]) for c in categories]
    raw_scores = np.array([SENTIMENT_SCORES[c] for c in categories])

    # Price = responds to true sentiment with `sentiment_lead` day lag + own momentum
    price    = np.zeros(n_days)
    price[0] = 100.0
    for i in range(1, n_days):
        sent_effect = true_sent[max(0, i - sentiment_lead)] * 0.003
        momentum    = 0.0002
        noise       = 0.010 * rng.standard_normal()
        ret         = momentum + sent_effect + noise
        price[i]    = price[i-1] * np.exp(np.clip(ret, -0.12, 0.12))

    df = pd.DataFrame({
        "price":      price,
        "true_sent":  true_sent,
        "obs_sent":   obs_sent,
        "raw_score":  raw_scores,
        "headline":   headlines,
        "category":   categories,
    }, index=dates)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SENTIMENT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_sentiment(df, decay_halflife=5, smooth_window=10):
    """
    1. Apply exponential decay to sentiment (old news fades)
    2. Smooth with rolling mean to reduce noise
    3. Normalise to z-score for signal generation
    """
    # Exponential decay weighting (sentiment relevance fades)
    alpha      = 1 - np.exp(-np.log(2) / decay_halflife)
    df["sent_decay"] = df["raw_score"].ewm(alpha=alpha, adjust=False).mean()

    # Smooth
    df["sent_smooth"] = df["sent_decay"].rolling(smooth_window).mean()

    # Normalise
    roll_mu  = df["sent_smooth"].rolling(60).mean()
    roll_std = df["sent_smooth"].rolling(60).std()
    df["sent_zscore"] = (df["sent_smooth"] - roll_mu) / roll_std.clip(0.01)

    # Sentiment momentum (change in smoothed sentiment)
    df["sent_momentum"] = df["sent_smooth"].diff(5)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LEAD-LAG ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def lead_lag_analysis(df, max_lag=10):
    """
    Cross-correlate sentiment with future price returns.
    Positive correlation at lag k means sentiment leads price by k days.
    """
    price_ret = df["price"].pct_change().fillna(0).values
    sentiment = df["sent_smooth"].fillna(0).values

    lags  = range(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag > 0:
            corr = np.corrcoef(sentiment[:-lag], price_ret[lag:])[0, 1]
        elif lag < 0:
            corr = np.corrcoef(sentiment[-lag:], price_ret[:lag])[0, 1]
        else:
            corr = np.corrcoef(sentiment, price_ret)[0, 1]
        corrs.append(corr)

    return list(lags), corrs


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STRATEGY SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

def signal_sentiment_only(df, z_entry=0.8, z_exit=0.0):
    """Long when sentiment z-score > entry, exit when drops below exit."""
    sig = pd.Series(0, index=df.index)
    pos = False
    for i in range(len(df)):
        z = df["sent_zscore"].iloc[i]
        if np.isnan(z): continue
        if not pos and z > z_entry:  pos = True
        elif pos and z < z_exit:     pos = False
        sig.iloc[i] = 1 if pos else 0
    return sig


def signal_price_only(df, fast=10, slow=30):
    """Simple price momentum: long when fast MA > slow MA."""
    ma_f = df["price"].rolling(fast).mean()
    ma_s = df["price"].rolling(slow).mean()
    sig  = pd.Series(0, index=df.index)
    sig[ma_f > ma_s] = 1
    return sig


def signal_combined(df, z_entry=0.5, fast=10, slow=30):
    """
    Combined: require BOTH positive sentiment AND price momentum.
    Higher bar = fewer but higher quality trades.
    """
    sent_sig  = signal_sentiment_only(df, z_entry=z_entry)
    price_sig = signal_price_only(df, fast=fast, slow=slow)
    return (sent_sig & price_sig).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_signal(df, signal, label="Strategy"):
    """Standard equity-curve backtest."""
    price  = df["price"]
    cost   = COMMISSION + SLIPPAGE
    equity = [INITIAL]
    cash   = INITIAL; pos = 0; entry_p = 0; trades = []

    for i in range(1, len(price)):
        sig  = signal.iloc[i-1]
        p    = price.iloc[i]

        if sig == 1 and pos == 0:
            exec_p = p * (1 + cost); pos = cash / exec_p
            cash = 0; entry_p = exec_p
        elif sig == 0 and pos > 0:
            exec_p = p * (1 - cost)
            trades.append((exec_p - entry_p) / entry_p)
            cash = pos * exec_p; pos = 0

        equity.append(cash + pos * p)

    eq = pd.Series(equity, index=df.index)
    ret = eq.pct_change().dropna()
    dd  = (eq - eq.cummax()) / eq.cummax()
    years = len(eq) / 252
    cagr  = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    pf     = sum(wins) / abs(sum(losses)) if losses else np.inf

    return {
        "label": label, "equity": eq, "drawdown": dd, "trades": trades,
        "metrics": {
            "Total Return (%)": round((eq.iloc[-1]/INITIAL-1)*100, 2),
            "CAGR (%)":         round(cagr*100, 2),
            "Sharpe Ratio":     round(sharpe, 3),
            "Max Drawdown (%)": round(dd.min()*100, 2),
            "Profit Factor":    round(pf, 3),
            "Win Rate (%)":     round(len(wins)/len(trades)*100, 2) if trades else 0,
            "# Trades":         len(trades),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EVENT STUDY
# ═══════════════════════════════════════════════════════════════════════════════

def event_study(df, threshold=1.5, window=10):
    """
    Average price return around strong sentiment events.
    Shows whether sentiment spikes predict future returns.
    """
    events  = df.index[df["raw_score"].abs() > threshold]
    pos_ev  = df.index[df["raw_score"] >  threshold]
    neg_ev  = df.index[df["raw_score"] < -threshold]
    prices  = df["price"].values

    def avg_path(event_dates, direction="pos"):
        paths = []
        for d in event_dates:
            idx = df.index.get_loc(d)
            if idx < window or idx + window >= len(prices): continue
            path = prices[idx-window:idx+window+1]
            path = path / path[window]   # normalise at event
            if direction == "neg":
                path = 2 - path          # flip for negative events
            paths.append(path)
        return np.array(paths).mean(axis=0) if paths else None

    pos_path = avg_path(pos_ev, "pos")
    neg_path = avg_path(neg_ev, "neg")
    days     = np.arange(-window, window+1)
    return days, pos_path, neg_path


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TEARSHEET
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tearsheet(save_path="/home/claude/tearsheet_sentiment.png"):
    df  = simulate_sentiment_and_price()
    df  = process_sentiment(df)

    sig_s = signal_sentiment_only(df)
    sig_p = signal_price_only(df)
    sig_c = signal_combined(df)

    bt_s  = backtest_signal(df, sig_s, "Sentiment Only")
    bt_p  = backtest_signal(df, sig_p, "Price Momentum")
    bt_c  = backtest_signal(df, sig_c, "Combined")
    bh_eq = (df["price"] / df["price"].iloc[0]) * INITIAL

    lags, corrs = lead_lag_analysis(df)
    evdays, pos_path, neg_path = event_study(df)

    fig = plt.figure(figsize=(22, 30), facecolor=DARK)
    gs  = gridspec.GridSpec(5, 2, figure=fig,
                            hspace=0.52, wspace=0.35,
                            top=0.93, bottom=0.04,
                            left=0.07, right=0.97)

    def sa(ax):
        ax.set_facecolor(CARD); ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[:].set_color(GRID)
        ax.grid(True, color=GRID, lw=0.4, ls="--", alpha=0.5)

    tk = dict(color=LIGHT, fontweight="bold", fontsize=10)

    fig.text(0.5, 0.965, "PROJECT 3 — SENTIMENT + PRICE MODEL",
             ha="center", fontsize=18, fontweight="bold",
             color="#f8fafc", fontfamily="monospace")
    fig.text(0.5, 0.950,
             "NLP Sentiment Scoring  |  Decay Model  |  Lead-Lag Analysis  |  Event Study  |  Combined Backtest",
             ha="center", fontsize=10, color=MUTED, fontfamily="monospace")

    # ── 1. Price + Sentiment overlay ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    sa(ax1); ax1.set_title("Price vs Smoothed Sentiment Signal", **tk)
    ax1b = ax1.twinx()

    ax1.plot(df.index, df["price"], color=ACCENT, lw=1.8, label="Price", zorder=2)
    ax1b.fill_between(df.index, 0, df["sent_smooth"],
                      where=df["sent_smooth"] > 0, color=GREEN, alpha=0.35)
    ax1b.fill_between(df.index, 0, df["sent_smooth"],
                      where=df["sent_smooth"] < 0, color=RED, alpha=0.35)
    ax1b.plot(df.index, df["sent_smooth"], color=AMBER, lw=1.2,
              alpha=0.8, label="Sentiment (smoothed)")
    ax1b.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax1.set_ylabel("Price ($)", color=ACCENT)
    ax1b.set_ylabel("Sentiment Score", color=AMBER)
    ax1b.tick_params(colors=MUTED, labelsize=8)
    ax1b.spines[:].set_color(GRID)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)
    ax1b.legend(loc="upper right", fontsize=8, framealpha=0.2, facecolor=CARD,
                edgecolor=GRID, labelcolor=LIGHT)

    # ── 2. Sentiment Z-score ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    sa(ax2); ax2.set_title("Sentiment Z-Score — Signal Generation", **tk)
    z = df["sent_zscore"]
    ax2.plot(df.index, z, color=PURPLE, lw=1.2, alpha=0.85)
    ax2.fill_between(df.index, z, 0, where=z>0, color=GREEN, alpha=0.25)
    ax2.fill_between(df.index, z, 0, where=z<0, color=RED,   alpha=0.25)
    ax2.axhline( 0.8, color=GREEN, lw=1.5, ls="--", label="Entry threshold")
    ax2.axhline(-0.8, color=RED,   lw=1.5, ls="--")
    ax2.axhline(0,    color=MUTED, lw=0.8)
    ax2.set_ylabel("Sentiment Z-score", color=MUTED)
    ax2.legend(fontsize=8, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)

    # ── 3. Lead-lag ──────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    sa(ax3); ax3.set_title("Lead-Lag: Does Sentiment Lead Price?", **tk)
    colors_ll = [GREEN if c > 0 else RED for c in corrs]
    ax3.bar(lags, corrs, color=colors_ll, alpha=0.8, zorder=3)
    ax3.axvline(0, color=AMBER, lw=1.5, ls="--")
    ax3.axhline(0, color=MUTED, lw=0.8)
    ax3.set_xlabel("Lag (days) — positive = sentiment leads price", color=MUTED)
    ax3.set_ylabel("Correlation", color=MUTED)
    best_lag = lags[np.argmax(corrs)]
    ax3.text(best_lag, max(corrs)*0.85, f"Peak lag\n= {best_lag}d",
             color=AMBER, fontsize=8, ha="center")

    # ── 4. Event study ───────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    sa(ax4); ax4.set_title("Event Study — Avg Price Around Sentiment Spikes", **tk)
    if pos_path is not None:
        ax4.plot(evdays, (pos_path - 1)*100, color=GREEN, lw=2.5,
                 label="After positive sentiment spike")
    if neg_path is not None:
        ax4.plot(evdays, (neg_path - 1)*100, color=RED, lw=2.5,
                 label="After negative sentiment spike")
    ax4.axvline(0, color=AMBER, lw=1.5, ls="--", label="Event day")
    ax4.axhline(0, color=MUTED, lw=0.8)
    ax4.set_xlabel("Days relative to event", color=MUTED)
    ax4.set_ylabel("Avg cumulative return (%)", color=MUTED)
    ax4.legend(fontsize=8, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)

    # ── 5. Equity curves ─────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    sa(ax5); ax5.set_title("Strategy Comparison — Equity Curves", **tk)
    ax5.plot(bh_eq.index, bh_eq, color=MUTED,   lw=1.5, ls="--", label="Buy & Hold")
    ax5.plot(bt_p["equity"].index, bt_p["equity"], color=RED,    lw=2,   label="Price Momentum")
    ax5.plot(bt_s["equity"].index, bt_s["equity"], color=AMBER,  lw=2,   label="Sentiment Only")
    ax5.plot(bt_c["equity"].index, bt_c["equity"], color=GREEN,  lw=2.5, label="Combined (Sentiment + Price)")
    ax5.set_ylabel("Portfolio Value ($)", color=MUTED)
    ax5.legend(fontsize=9, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)

    # ── 6. Metrics table ─────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[4, :])
    ax6.set_facecolor(CARD); ax6.axis("off")
    ax6.set_title("Performance Metrics — Net of Commission & Slippage",
                  color=LIGHT, fontweight="bold", fontsize=10)

    bh_ret = (bh_eq.iloc[-1]/INITIAL - 1)*100
    bh_dd  = ((bh_eq - bh_eq.cummax())/bh_eq.cummax()).min()*100
    bh_sh  = bh_eq.pct_change().dropna()
    bh_sh  = bh_sh.mean()/bh_sh.std()*np.sqrt(252)

    rows = [
        ["Buy & Hold",      f"{bh_ret:.1f}%",  "—",    f"{bh_sh:.2f}", f"{bh_dd:.1f}%",  "—",    "—",   "1"],
        ["Price Momentum",  f"{bt_p['metrics']['Total Return (%)']:.1f}%",
                            f"{bt_p['metrics']['CAGR (%)']:.1f}%",
                            f"{bt_p['metrics']['Sharpe Ratio']:.2f}",
                            f"{bt_p['metrics']['Max Drawdown (%)']:.1f}%",
                            f"{bt_p['metrics']['Profit Factor']:.2f}",
                            f"{bt_p['metrics']['Win Rate (%)']:.1f}%",
                            f"{bt_p['metrics']['# Trades']}"],
        ["Sentiment Only",  f"{bt_s['metrics']['Total Return (%)']:.1f}%",
                            f"{bt_s['metrics']['CAGR (%)']:.1f}%",
                            f"{bt_s['metrics']['Sharpe Ratio']:.2f}",
                            f"{bt_s['metrics']['Max Drawdown (%)']:.1f}%",
                            f"{bt_s['metrics']['Profit Factor']:.2f}",
                            f"{bt_s['metrics']['Win Rate (%)']:.1f}%",
                            f"{bt_s['metrics']['# Trades']}"],
        ["Combined ★",      f"{bt_c['metrics']['Total Return (%)']:.1f}%",
                            f"{bt_c['metrics']['CAGR (%)']:.1f}%",
                            f"{bt_c['metrics']['Sharpe Ratio']:.2f}",
                            f"{bt_c['metrics']['Max Drawdown (%)']:.1f}%",
                            f"{bt_c['metrics']['Profit Factor']:.2f}",
                            f"{bt_c['metrics']['Win Rate (%)']:.1f}%",
                            f"{bt_c['metrics']['# Trades']}"],
    ]
    cols = ["Strategy","Total Return","CAGR","Sharpe","Max DD","Profit Factor","Win Rate","# Trades"]
    tbl  = ax6.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 2.0)
    for (r2, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r2 == 0:
            cell.set_facecolor("#1e3a5f")
            cell.set_text_props(color="#93c5fd", fontweight="bold")
        elif r2 == 4:
            cell.set_facecolor("#14532d")
            cell.set_text_props(color="#4ade80", fontweight="bold")
        else:
            cell.set_facecolor("#1e293b" if r2%2==0 else "#172032")
            cell.set_text_props(color=LIGHT)

    plt.savefig(save_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [✓] Sentiment tearsheet → {save_path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PROJECT 3: SENTIMENT + PRICE MODEL")
    print("="*60)
    df  = simulate_sentiment_and_price()
    df  = process_sentiment(df)

    sig_c = signal_combined(df)
    bt_c  = backtest_signal(df, sig_c, "Combined")
    m = bt_c["metrics"]
    print(f"\n  Combined Strategy Results:")
    for k, v in m.items():
        tag = "  ✅ BEATS TARGET" if k=="Profit Factor" and isinstance(v,float) and v>=1.5 else ""
        print(f"    {k:<25} {str(v):>10}{tag}")

    print("\n  Building tearsheet...")
    plot_tearsheet()
