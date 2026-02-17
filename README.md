# Statistical Arbitrage — High-Frequency Pairs Trading

A pairs trading framework built from first principles on 10-second order book data, developed independently in 72 hours. The project covers the full research pipeline: from microstructure-aware data cleaning through cointegration analysis, dynamic hedge ratio estimation, and a complete backtesting engine with realistic execution assumptions.

---

## Motivation

Standard pairs trading tutorials assume clean price series and ignore market microstructure entirely. This project takes the opposite approach — starting from the raw order book and asking: *what does it actually mean for two instruments to be in equilibrium, and how do you detect and trade deviations from it in a high-frequency setting?*

The two instruments exhibit persistent co-movement across most trading regimes, making them a natural candidate for a mean-reversion strategy. The key challenge is doing the statistical analysis rigorously rather than naively — distinguishing genuine cointegration from spurious correlation, and building execution logic that respects the bid-ask spread rather than assuming trades occur at midprice.

---

## Methodology

### 1. Data Cleaning & Microstructure Handling

Raw 10-second snapshots require significant preprocessing before any statistical analysis is valid:

- **Crossed market detection** — rows where `ASK ≤ BID` indicate executed trades and are removed to preserve a clean quote-only dataset
- **Weekend and public holiday removal** — weekends exhibit near-zero volume and stale quotes; keeping them contaminates rolling statistics with unrepresentative observations
- **Inactive market detection** — contiguous periods where all quote fields are frozen (≥ 1 minute) are identified and removed; these represent pre-market, post-market, or halted trading windows
- **Overnight gap handling** — log return calculations respect session boundaries so overnight gaps do not produce spurious large returns

### 2. Microprice Construction

Rather than using naive midprice, a **Volume-Weighted Average Price (VWAP)** microprice is constructed:

$$X_t = \frac{ASK_t \cdot ASK\_VOL_t + BID_t \cdot BID\_VOL_t}{ASK\_VOL_t + BID\_VOL_t}$$

This accounts for order book imbalances — when bid volume significantly exceeds ask volume, buying pressure is dominant and the fair value sits closer to the ask. Using midprice in these conditions introduces a systematic bias into spread estimation.

### 3. Stationarity Analysis

Before testing cointegration, stationarity of the individual price series is established from first principles. Starting from the AR(1) process:

$$X_t = \mu + \phi X_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim (0, \sigma^2)$$

Recursive substitution shows:

$$\mathbb{E}[X_t] = \mu + \phi^t X_0, \qquad \text{Var}(X_t) = \sigma^2 \sum_{k=0}^{t-1} \phi^{2k}$$

Stationarity requires $|\phi| < 1$; when $\phi = 1$ the process is a random walk with time-varying mean and variance — the unit root case. The **Augmented Dickey-Fuller (ADF)** test formalises this:

$$\Delta X_t = \mu + \delta X_{t-1} + \sum_{i=1}^{P} \beta_i \Delta X_{t-i} + \varepsilon_t$$

The null hypothesis is $\delta = 0$ (unit root present). ACF/PACF analysis of the differenced spread guided the lag selection of $P = 5$. Both instruments fail to reject the unit root null at conventional significance levels, confirming they are I(1) processes.

### 4. Cointegration Testing

With both series confirmed as I(1), the Engle-Granger procedure tests whether a stationary linear combination exists. The hedge ratio is estimated via OLS in both directions (X on Y, Y on X), with the direction yielding the more negative ADF statistic selected. The **Engle-Granger test** with MacKinnon-adjusted critical values is then applied to the residual:

$$S_t = Y_t - (\hat{\alpha} + \hat{\beta} X_t)$$

Results indicate **weak cointegration** — the null of no cointegration is rejected at the 10% level but not the 5% level. This motivates a conservative strategy design that does not assume a strongly stable long-run equilibrium, and reinforces the need for a rolling (rather than static) hedge ratio.

> **Note on spurious regression:** Direct OLS on price levels produces very high $R^2$ driven by shared non-stationary drift — not genuine co-movement. All inference is correctly performed on residuals and returns, not levels.

### 5. Dynamic Hedge Ratio via Rolling OLS

A static hedge ratio is insufficient given the observed slow-moving drift in the spread mean across the sample. A **10-day rolling OLS** window estimates the time-varying parameters $(\alpha_t, \beta_t)$:

$$S_t = Y_t - (\alpha_t + \beta_t X_t)$$

The 10-day window reflects a deliberate bias-variance tradeoff: short enough to adapt to medium-term regime shifts, long enough to prevent noisy parameter estimates from microstructure effects.

### 6. Time-Decay EMA Smoothing

Before constructing the spread, raw microprices are smoothed using a **time-decay Exponential Moving Average** that accounts for irregular time gaps in the data:

$$\alpha_i = e^{-\lambda \cdot \Delta t_i}, \qquad \lambda = \frac{\ln 2}{\text{halflife}}$$

This is critical for two reasons: (1) standard span-based EMAs assume uniform time steps, which is violated by the inactive-market gaps removed during cleaning; (2) smoothing attenuates bid-ask bounce and transient liquidity shocks that would otherwise inflate spread variance and generate false signals.

### 7. Trading Strategy

Entry and exit signals are based on the rolling z-score of the spread:

$$z_t = \frac{S_t - \mu_t}{\sigma_t}$$

| Signal | Condition | Action |
|--------|-----------|--------|
| Long spread | $z_t < -1.5$ | Buy $Y$, sell $\beta_t$ units of $X$ |
| Short spread | $z_t > +1.5$ | Sell $Y$, buy $\beta_t$ units of $X$ |
| Exit | $\|z_t\| < 0.5$ | Close position |

The $\pm 1.5\sigma$ threshold corresponds to approximately 6.7% one-sided tail probability under Gaussianity, ensuring trades are triggered only on meaningful dislocations rather than noise. Execution uses bid prices for sells and ask prices for buys, incorporating transaction costs realistically.

---

## Future Work

**Order Book Imbalance (OBI) as a signal** — OBI measures relative buy/sell pressure at the top of book:

$$OBI_t = \frac{BID\_VOL_t - ASK\_VOL_t}{BID\_VOL_t + ASK\_VOL_t}$$

OBI captures short-term order flow pressure that is orthogonal to the spread z-score. A natural extension is a composite entry signal that requires both a z-score threshold crossing *and* a confirming OBI direction, reducing adverse selection on entry.

**Regime-aware strategy via Hidden Markov Model** — rolling correlation between the instruments exhibits clear distributional shifts, with the COVID-19 volatility shock (March–April 2020) producing a distinct high-correlation, high-spread regime. An HMM over observable features (rolling volatility, rolling correlation, bid-ask spread) could identify the latent market regime and allow the strategy to adjust thresholds or pause trading in unfavourable regimes. This draws on multi-target tracking methodology where similar state-space approaches are used for regime identification.

**Kalman filter for hedge ratio estimation** — rolling OLS is a reasonable baseline but treats the hedge ratio as piecewise-constant within each window. A Kalman filter would model $\beta_t$ as a continuously evolving latent state, producing smoother, more adaptive estimates with principled uncertainty quantification.

---

## Dependencies

```
numpy
pandas
statsmodels
matplotlib
seaborn
```

---

## Structure

```
PairsTrading.ipynb    # Full analysis notebook
final_data_10s.csv    # 10-second order book snapshots (not included)
```