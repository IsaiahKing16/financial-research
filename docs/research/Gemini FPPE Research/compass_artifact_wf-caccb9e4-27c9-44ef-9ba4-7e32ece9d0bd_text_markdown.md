# Open-source projects resembling KNN-based financial analogue prediction

**No single public repository replicates System A's full methodology** — the combination of KNN on multi-day return fingerprints, cross-ticker analogue matching across 25 years, probability calibration, and regime conditioning appears genuinely novel in the open-source ecosystem. However, **27 repositories and libraries** address individual components or closely related approaches. The most relevant is `chinuy/stock-price-prediction`, which independently validates the multi-day return fingerprint concept and found that KNN specifically benefits from **99 days** of return history as features — far more than tree-based or neural methods need.

What follows is a tiered inventory of every relevant project found, organized from most to least similar to System A's architecture.

---

## Direct analogue-matching systems for stock prediction

These repos implement the core concept: find historically similar moments, then predict based on what happened next.

### chinuy/stock-price-prediction ★ Most relevant overall

- **URL:** https://github.com/chinuy/stock-price-prediction
- **Stars:** 20 | **Forks:** 13 | **License:** MIT | **Language:** Python
- **Last activity:** Inactive (~2017–2018)
- **What it does:** Transforms next-day prediction into binary Up/Down classification using multi-day return features. Tests KNN, SVM, Random Forest, and RNN across **9 sector ETFs** (XLE, XLU, XLK, XLB, XLP, XLY, XLI, XLV, SPY). The `delta` parameter controls how many days of OHLC, volume, returns, close-price percentage change, and rolling mean of returns form the feature vector.
- **Relation to System A:** This is the closest conceptual match found. It uses **multi-day return windows as KNN features** — exactly the return fingerprint concept. Cross-validated the `delta` parameter and found **KNN's optimal delta is 99 days**, far exceeding SVM (4 days) or Random Forest (3 days). This directly validates System A's approach of using extended return windows for nearest-neighbor matching. Uses **expanding-window time-series cross-validation** (10-fold), achieving a KNN Sharpe ratio of **0.990** — competitive with SVM's 1.019.
- **Key differences:** Only 9 ETFs (not 52 tickers), 2 years of data (not 25), next-day prediction (not 7-day), no probability calibration, no regime conditioning, no Brier Skill Score, no ball_tree specification, no autonomous pipeline.

### DayuanTan/knn_predictprice ★ Closest fingerprint architecture

- **URL:** https://github.com/DayuanTan/knn_predictprice
- **Stars:** 1 | **Forks:** 0 | **Language:** Python
- **Last activity:** Completed project (small)
- **What it does:** Predicts stock prices using a nearest-neighbor search over **8 consecutive daily price changes** (open, high, low, close changes). Finds the single closest historical match (k=1) from a dataset spanning **5 years across multiple companies**, then uses that match's subsequent price change to predict the next day.
- **Relation to System A:** Architecturally the most similar — uses an **8-feature price-change fingerprint** (matching System A's 8-feature design), Euclidean distance, and cross-stock historical search. The companion repo `DayuanTan/prepare_data_4knn` handles multi-stock data preparation.
- **Key differences:** Uses k=1 (not probabilistic K-nearest), consecutive daily changes (not multi-period return windows like 1d/2d/5d/10d), 5 years of data (not 25), no calibration, no regime conditioning, no walk-forward validation, no Brier Skill Score.

### nsarang/big-data-stock-price-forecast ★ VAE-encoded analogue matching

- **URL:** https://github.com/nsarang/big-data-stock-price-forecast
- **Stars:** 1 | **Forks:** 0 | **Language:** Jupyter Notebook (94.7%), Python
- **Last activity:** Completed project (~12 commits)
- **What it does:** Encodes **256-hour OHLCV windows** into a latent space via a Variational Autoencoder (PyTorch), then performs similarity search using cosine, Euclidean, and L1 distance to find the best historical match. Predicts by re-scaling the matched pattern's subsequent 128-hour trajectory.
- **Relation to System A:** Same conceptual framework — find historically similar moment → predict based on what happened next. The VAE embedding replaces System A's handcrafted return fingerprint with learned representations. Tests multiple distance metrics.
- **Key differences:** Crypto only (BTC, ETH, LTC, XRP), hourly data (not daily), deep learning encoding (not return-based features), averages only 2 best matches (not K neighbors), no probability output, no regime conditioning.

### mason-lee19/DtwStockAnalysis ★ DTW-based analogue matching

- **URL:** https://github.com/mason-lee19/DtwStockAnalysis
- **Stars:** 2 | **Language:** Python 100%
- **Last activity:** 5 commits (small project)
- **What it does:** Uses Dynamic Time Warping to compare current n-day stock windows (tested 20–90 days) against all historical windows. If DTW cost falls below a threshold, records the subsequent day's movement. Backtested on **12 S&P 500 companies**.
- **Relation to System A:** Nearly identical framework — sliding window comparison against historical patterns, predict based on matched outcomes. Converts to percent change from window start (similar to returns normalization). Found optimal window of **50 days** with cost threshold 0.5.
- **Key differences:** Uses DTW (not Euclidean on return fingerprints), single-ticker matching (not cross-ticker), next-day only (not 7-day), threshold-based matching (not K-nearest). Author's honest conclusion: "this does not look to be a profitable strategy" — though lack of calibration and regime conditioning may explain this.

---

## Pattern similarity search tools and libraries

These projects provide similarity-based pattern discovery infrastructure applicable to financial time series.

### stumpy-dev/stumpy ★ Matrix profile engine (by TD Ameritrade)

- **URL:** https://github.com/stumpy-dev/stumpy
- **Stars:** ~3,400+ | **License:** BSD-3 | **Actively maintained** (v1.14+)
- **What it does:** Computes matrix profiles — for every subsequence in a time series, finds its nearest neighbor using z-normalized Euclidean distance. Created by **TD Ameritrade** (now Charles Schwab). The `stumpy.match()` function finds all subsequences similar to a query pattern. Supports GPU acceleration and multi-dimensional time series.
- **Relation to System A:** The **AB-join** functionality is designed exactly for cross-ticker pattern matching — finding conserved patterns between two independent time series. The library's stock pattern matching tutorial explicitly describes finding historical trading patterns across ticker symbols. Could serve as System A's core similarity engine at scale.
- **Key differences:** Infrastructure library, not a prediction system. No classification output, no calibration, no regime conditioning. Uses z-normalized Euclidean distance rather than raw return fingerprints.

### tslearn-team/tslearn ★ Time series KNN classifier

- **URL:** https://github.com/tslearn-team/tslearn
- **Stars:** ~2,900+ | **Actively maintained** (v0.7.0, Nov 2025)
- **What it does:** Full-featured Python toolkit for time series ML. Includes `KNeighborsTimeSeriesClassifier` with DTW, Euclidean, and SoftDTW metrics. scikit-learn compatible (works with GridSearchCV, pipelines).
- **Relation to System A:** Provides a ready-made KNN classifier specifically designed for time series data. Could directly replace System A's custom KNN implementation with DTW-aware distance metrics that handle temporal misalignment between return patterns.
- **Key differences:** General-purpose library, not finance-specific. No financial feature engineering, no calibration layer, no regime conditioning.

### gaborvecsei/Stocks-Pattern-Analyzer ★ Most starred pattern search tool

- **URL:** https://github.com/gaborvecsei/Stocks-Pattern-Analyzer
- **Stars:** 259 | **Forks:** 86 | **Language:** Python (73.7%)
- **What it does:** Deployable web application (REST API + Plotly Dash frontend) for discovering similar historical patterns in stock data. Supports S&P 500 symbols and currency pairs. Docker/Heroku deployable.
- **Relation to System A:** Similarity-based pattern search across S&P 500, designed as a usable tool rather than a notebook exercise.
- **Key differences:** Exploratory/visual tool rather than a predictive system. No formalized KNN, no probability output, no automated pipeline.

### facebookresearch/faiss ★ Scalable nearest-neighbor infrastructure

- **URL:** https://github.com/facebookresearch/faiss
- **Stars:** ~37,000+ | **Actively maintained**
- **What it does:** Library for efficient similarity search of dense vectors. Supports L2 and dot product distances with multiple index types (brute force, IVF, PQ, HNSW). Scales to **1.5 trillion vectors** at Meta.
- **Relation to System A:** If scaling from 52 to 5,200 tickers (as System A plans), the fingerprint database grows from ~260K to ~26M vectors. FAISS's IVF+PQ indexes would enable sub-millisecond queries at that scale. No financial prediction repos currently use FAISS — this represents an open opportunity.
- **Key differences:** Pure infrastructure. For System A's current 52-ticker scale (~260K vectors), sklearn's `ball_tree` is sufficient. FAISS becomes relevant at 5,200+ tickers.

---

## Regime detection and macro conditioning

These repos address System A's regime-aware prediction component.

### tubakhxn/Market-Regime-Detection-System ★ Closest to System A's regime approach

- **URL:** https://github.com/tubakhxn/Market-Regime-Detection-System
- **Stars:** 5 | **Forks:** 2 | **License:** MIT | **Language:** Python
- **What it does:** Classifies market regimes as **Bull, Bear, and Sideways using SPY data** with ML and technical indicators.
- **Relation to System A:** Most directly aligned — specifically classifies bull/bear/sideways from SPY, exactly like System A's SPY 90-day return classifier. The three-regime classification (adding "sideways") is an extension of System A's binary approach.
- **Key differences:** Uses ML classifiers on technical indicators rather than a simple 90-day return threshold. More complex but potentially less interpretable.

### Sakeeb91/market-regime-detection ★ HMM-based with walk-forward

- **URL:** https://github.com/Sakeeb91/market-regime-detection
- **Stars:** 1 | **Language:** Python
- **What it does:** Detects bull, bear, and high-volatility regimes using Hidden Markov Models (HMM) with Gaussian emissions and Gaussian Mixture Models. Includes regime-conditioned trading strategies and walk-forward validation. Uses **SPY** data from Yahoo Finance.
- **Relation to System A:** Uses SPY data, classifies regimes, conditions trading strategy on detected regime, implements walk-forward validation — all present in System A. The HMM approach auto-discovers regime boundaries rather than using a fixed lookback window.
- **Key differences:** HMM-based (not simple return threshold), includes a high-volatility third regime, no KNN prediction component.

### atkrish0/adverse-regime-detection ★ Macro-indicator approach

- **URL:** https://github.com/atkrish0/adverse-regime-detection
- **Stars:** Small | **Language:** Python
- **What it does:** Forecasts adverse regimes using **129 leading macroeconomic indicators** from the FRED dataset (McCracken & Ng, 2015). Creates 710 features with lags, uses TimeSeriesSplit CV, labels regimes using NBER recession dates.
- **Relation to System A:** The most "macro-conditioned" approach found. While System A uses a single macro signal (SPY 90-day return), this repo demonstrates the rich feature space available for regime detection using economic fundamentals.
- **Key differences:** Recession-focused rather than bull/bear classification, uses macro indicators (not market returns), no stock-level prediction component.

### QuhiQuhihi/regime_model

- **URL:** https://github.com/QuhiQuhihi/regime_model
- **Stars:** ~30+ | **Language:** Python
- **What it does:** Gaussian Mixture Model and HMM for detecting regime shifts across stocks, bonds, real estate, and commodities. Inspired by Two Sigma's "A Machine Learning Approach to Regime Modeling."
- **Relation to System A:** Multi-asset regime detection with solid statistical foundation. Broader asset class coverage than System A.
- **Key differences:** Portfolio allocation focus, not prediction conditioning. No KNN component.

---

## KNN stock prediction implementations

These repos use KNN for stock prediction but with simpler feature engineering than System A.

### kennedyCzar/STOCK-RETURN-PREDICTION-USING-KNN-SVM-...

- **URL:** https://github.com/kennedyCzar/STOCK-RETURN-PREDICTION-USING-KNN-SVM-GUASSIAN-PROCESS-ADABOOST-TREE-REGRESSION-AND-QDA
- **Stars:** 137 | **Forks:** 35 | **Language:** Python
- **What it does:** Compares KNN (best k=10, **77.9% accuracy**), Random Forest (85.5%), SVM, Gaussian Process, AdaBoost, Gradient Boosting, and QDA for stock return prediction. Uses sklearn Pipeline + GridSearchCV. Tests across AAPL, IBM, GOLD, and others. Finds different algorithms win for different stocks.
- **Relation to System A:** Demonstrates KNN as a competitive stock return predictor with proper hyperparameter tuning. The per-stock algorithm variation finding is interesting — System A's cross-ticker approach may mitigate this.
- **Key differences:** Technical indicators as features (not return fingerprints), no cross-ticker matching, no calibration, no regime conditioning.

### sammanthp007/Stock-Price-Prediction-Using-KNN-Algorithm

- **URL:** https://github.com/sammanthp007/Stock-Price-Prediction-Using-KNN-Algorithm
- **Stars:** 82 | **Forks:** 55 | **Language:** Python
- **What it does:** Custom KNN implementation from scratch using Euclidean distance for 6 NASDAQ companies. Features: OHLCV + percent change. Achieves ~70% directional accuracy. Includes research paper (`final.pdf`).
- **Relation to System A:** Pure KNN with Euclidean distance for stock direction prediction — same core algorithm family. Custom implementation allows inspection of the distance calculation.
- **Key differences:** Single-day OHLCV features (not multi-day return windows), single-ticker, no walk-forward validation, no calibration.

### eiahb3838ya/PHBS_ML_for_quant_project

- **URL:** https://github.com/eiahb3838ya/PHBS_ML_for_quant_project
- **Stars:** Small academic project | **Language:** Python
- **What it does:** Expanding-window training with KNN and other classifiers for predicting Wind All A Index. Uses **52 daily factors + 8 WorldQuant101 alpha factors**. Retrains every 20 trading days on ≥1,800 days of expanding training data.
- **Relation to System A:** Demonstrates expanding-window walk-forward validation with KNN using a rich factor feature set. The 52-factor + 8-alpha setup is similar in spirit to System A's 8-feature fingerprint, and the 1,800+ day training window matches System A's long-history approach.
- **Key differences:** Factor-based features (not return fingerprints), index-level prediction (not individual stocks), Chinese market data.

---

## Calibration and evaluation tools

No repository was found applying probability calibration specifically to stock prediction outputs — this is a genuine gap in the open-source landscape.

### zygmuntz/classifier-calibration

- **URL:** https://github.com/zygmuntz/classifier-calibration
- **Stars:** ~150+ | **Language:** Python
- **What it does:** Clean implementations of Platt scaling (`platts_scaling.py`), isotonic regression (`isotonic_regression.py`), and reliability diagrams. Companion code for the well-known FastML blog.
- **Relation to System A:** Provides exactly the calibration building blocks (Platt + isotonic + reliability diagrams) that System A applies to KNN outputs. General-purpose but directly applicable.

### flimao/briercalc

- **URL:** https://github.com/flimao/briercalc
- **Stars:** Small | **Language:** Python
- **What it does:** Calculates **Brier scores, Brier Skill Scores**, calibration, resolution, and uncertainty decomposition for multiple classes.
- **Relation to System A:** Implements the exact evaluation metric (BSS > 0 = beats base rate) that System A uses. Includes the Murphy decomposition into calibration, resolution, and uncertainty — useful for diagnosing where prediction skill comes from.

### kernc/backtesting.py ★ KNN walk-forward tutorial

- **URL:** https://github.com/kernc/backtesting.py
- **Stars:** ~4,000–5,000 | **Actively maintained**
- **What it does:** Major backtesting library. Its official [ML trading tutorial](https://kernc.github.io/backtesting.py/doc/examples/Trading%20with%20Machine%20Learning.html) demonstrates `KNeighborsClassifier(7)` with **walk-forward retraining** every 20 iterations on 400-value windows. Includes commission modeling, stop-loss/take-profit.
- **Relation to System A:** Best public example of KNN + walk-forward validation for trading. The k=7 neighbors, walk-forward retraining cadence, and direction classification are all analogous to System A's design.
- **Key differences:** Forex data (EUR/USD), not equities. No return fingerprints, no calibration layer, no regime conditioning.

---

## Comprehensive ML platforms with relevant components

### microsoft/qlib ★ Production-grade quant platform

- **URL:** https://github.com/microsoft/qlib
- **Stars:** 37,300 | **Forks:** 5,800 | **License:** MIT | **Actively maintained** (v0.9.7, Aug 2025)
- **What it does:** Microsoft Research's AI-oriented quant platform. Full pipeline: data acquisition → Alpha158/Alpha360 feature engineering → 20+ SOTA models → backtesting → portfolio optimization → order execution. Supports **auto daily data updates via crontab from Yahoo Finance**, covering 800+ stocks.
- **Relation to System A:** The only project matching System A's **autonomous scheduled pipeline** concept (crontab-based daily updates). Multi-ticker, walk-forward retraining, production architecture. Demonstrates what "scaling to 5,200 tickers" looks like in practice.
- **Key differences:** No KNN or distance-based methods in its model zoo. Focuses on deep learning and tree-based models. Adding a KNN analogue matcher to Qlib's infrastructure would effectively recreate System A at scale.

### huseinzol05/Stock-Prediction-Models

- **URL:** https://github.com/huseinzol05/Stock-Prediction-Models
- **Stars:** ~9,200 | **Forks:** ~3,000 | **Archived:** July 2023
- **What it does:** Collection of 30+ ML/DL models for stock forecasting (LSTM, GRU, Monte Carlo, GAN, RL agents, KNN, Adaboost, etc.) with trading bots and simulations.
- **Relation to System A:** Includes KNN implementations among the 30+ approaches. Useful as a benchmark reference for comparing KNN against other model families.

### liorsidi/sp500-stock-similarity-time-series

- **URL:** https://github.com/liorsidi/sp500-stock-similarity-time-series
- **Stars:** 99 | **Forks:** 24 | **Language:** Python
- **What it does:** Tests **5 similarity functions** (DTW, SAX, co-integration, Euclidean, Pearson) to find related stocks in the S&P 500, then enriches Random Forest and Gradient Boosting training data with similar-stock features. Walk-forward validation with 5 folds. Co-integration yielded best results: **0.55 accuracy, $19.78 profit** vs. baseline 0.52 accuracy, $6.60 profit. Includes full research paper.
- **Relation to System A:** The most methodologically rigorous similarity-based stock prediction project. Tests multiple distance metrics on S&P 500 daily data (2012–2017) for ~500 stocks. However, uses similarity to find *related stocks* (cross-sectional), not *similar historical moments* (temporal analogue matching).

### rmontagnin/Forex-and-Stock-Python-Pattern-Recognizer

- **URL:** https://github.com/rmontagnin/Forex-and-Stock-Python-Pattern-Recognizer
- **Stars:** 155 | **Forks:** 76 | **Archived:** March 2020
- **What it does:** Searches current price windows against all historical windows, shows matching patterns with predicted rise (green) or fall (red) outcomes. Uses percent-change similarity with configurable thresholds.
- **Relation to System A:** Same prediction logic — find similar historical patterns, average their outcomes. Uses percent change (a form of returns) for comparison.
- **Key differences:** Forex (GBPUSD) only, threshold-based matching (not KNN), archived/unmaintained.

---

## Relevant academic papers with code or findings

Several academic papers validate the KNN analogue approach, though most lack public code repositories:

**"Trend-Based K-Nearest Neighbor Algorithm in Stock Price Prediction"** (Atlantis Press) found that as the length of constructed time series increases, prediction error decreases — directly validating System A's use of extended return windows. Combined with `chinuy`'s finding that KNN's optimal delta is 99 days, there is **academic support for the claim that KNN benefits from longer fingerprints than other ML methods**.

**"Predicting stock trends through technical analysis and nearest neighbor classification"** (IEEE, 2009) combined KNN with technical analysis filters and generated considerably higher profits than buy-and-hold with minimal market exposure. The **marcuswang6/stock-top-papers** repo (https://github.com/marcuswang6/stock-top-papers) curates top stock prediction papers including **"Multi-period Learning for Financial Time Series Forecasting" (KDD 2025)**, directly relevant to multi-period return feature engineering.

---

## What the landscape tells us about System A's novelty

The comprehensive search across GitHub reveals that **System A occupies an essentially empty niche**. The table below maps each of System A's components against the best available open-source match:

| System A Component | Best Open-Source Match | Gap |
|---|---|---|
| Multi-day return fingerprints (8 features) | chinuy (multi-day delta features) | No repo uses layered return windows (1d/2d/5d/10d/20d etc.) |
| KNN ball_tree across 52 tickers × 25 years | DayuanTan (cross-stock, 8-change fingerprint) | No repo pools neighbors across tickers at this scale |
| 7-day direction classification | backtesting.py tutorial (2-day) | No repo uses 7-day forward horizon |
| Platt scaling + isotonic calibration | zygmuntz/classifier-calibration | **No repo applies calibration to stock KNN outputs** |
| SPY 90-day regime conditioning | tubakhxn/Market-Regime-Detection-System | No repo conditions KNN predictions on regime |
| Brier Skill Score evaluation | flimao/briercalc | **No stock prediction repo uses BSS** |
| Autonomous EOD pipeline | microsoft/qlib (crontab) | No KNN-based system has autonomous scheduling |
| Confidence threshold tuning (~0.60) | None found | Completely absent from public repos |

The probability calibration applied to financial KNN outputs and the Brier Skill Score evaluation for stock prediction are the two components with **zero** existing public implementations. These represent the most differentiating aspects of System A relative to the open-source landscape.

## Conclusion

The search uncovered **chinuy/stock-price-prediction** as the single most validating reference — its independent finding that KNN needs ~99 days of return data as features (vs. 3–4 days for tree-based methods) provides empirical support for System A's core design principle. **STUMPY** and **tslearn** offer production-grade similarity search infrastructure that could enhance System A's scalability. **Microsoft's Qlib** demonstrates the autonomous pipeline architecture at scale. For regime detection, **tubakhxn's Market-Regime-Detection-System** most closely mirrors System A's SPY-based approach.

The absence of any project combining these elements suggests that publishing System A (or components of it) would fill a genuine gap. The financial ML community has extensively explored tree-based and deep learning approaches but has largely overlooked the KNN analogue-matching paradigm with calibrated probability outputs — despite academic evidence that it works.